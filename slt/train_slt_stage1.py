"""
Sign Language Translation (SLT) - Stage 1
SHuBERT encoder + linear projection + ByT5 decoder
Pre-training on YouTube-ASL with weak labels (auto captions)
"""

import numpy as np
import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import wandb

sys.path.insert(0, '/scratch/project_465002480/YouTube_ASL/custom_python_libs')
sys.path.insert(0, '/scratch/project_465002480/SHuBERT/fairseq')
sys.path.insert(0, '/scratch/project_465002480/SHuBERT/fairseq/examples')

from examples.shubert.models.shubert import SHubertModel, SHubertConfig
from peft import LoraConfig, get_peft_model


# ==========================================
# Dataset
# ==========================================
class YouTubeASLSLTDataset(Dataset):
    def __init__(self, features_base, json_path, tokenizer,
                 max_frames=500, max_text_len=256):
        self.features_base = Path(features_base)
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.max_text_len = max_text_len
        self.samples = []

        print(f'Loading {json_path}...', flush=True)
        with open(json_path) as f:
            data = json.load(f)

        for video_id, video_data in data.items():
            for clip_name, clip_data in video_data.items():
                if clip_name == 'clip_order':
                    continue
                translation = clip_data.get('translation', '').strip()
                if not translation:
                    continue

                face_path  = self.features_base / 'face'  / f'{clip_name}_face.npy'
                lhand_path = self.features_base / 'lhand' / f'{clip_name}_left_hand.npy'
                rhand_path = self.features_base / 'rhand' / f'{clip_name}_right_hand.npy'
                body_path  = self.features_base / 'body'  / f'{clip_name}_body.npy'

                if not all([p.exists() for p in [face_path, lhand_path, rhand_path, body_path]]):
                    continue

                self.samples.append((
                    str(face_path), str(lhand_path),
                    str(rhand_path), str(body_path),
                    translation
                ))

        print(f'Loaded {len(self.samples)} samples', flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        face_path, lhand_path, rhand_path, body_path, text = self.samples[idx]

        face  = np.load(face_path).astype(np.float32)
        lhand = np.load(lhand_path).astype(np.float32)
        rhand = np.load(rhand_path).astype(np.float32)
        body  = np.load(body_path).astype(np.float32)

        T = min(face.shape[0], self.max_frames)
        face, lhand, rhand, body = face[:T], lhand[:T], rhand[:T], body[:T]

        face[np.any(face == -9999, axis=1)]   = 0
        lhand[np.any(lhand == -9999, axis=1)] = 0
        rhand[np.any(rhand == -9999, axis=1)] = 0
        body[np.any(body == -9999, axis=1)]   = 0

        encoded = self.tokenizer(
            text,
            max_length=self.max_text_len,
            truncation=True,
            padding=False,
            return_tensors='pt'
        )

        return {
            'face':      torch.FloatTensor(face),
            'lhand':     torch.FloatTensor(lhand),
            'rhand':     torch.FloatTensor(rhand),
            'body':      torch.FloatTensor(body),
            'input_ids': encoded['input_ids'].squeeze(0),
            'T': T,
        }


def collate_fn(batch):
    max_T    = max(b['T'] for b in batch)
    max_tlen = max(b['input_ids'].shape[0] for b in batch)
    B = len(batch)

    face_batch  = torch.zeros(B, max_T, 50)
    lhand_batch = torch.zeros(B, max_T, 42)
    rhand_batch = torch.zeros(B, max_T, 42)
    body_batch  = torch.zeros(B, max_T, 14)
    lengths     = torch.zeros(B, dtype=torch.long)
    label_batch = torch.full((B, max_tlen), -100, dtype=torch.long)
    decoder_input_batch = torch.zeros(B, max_tlen, dtype=torch.long)

    for i, b in enumerate(batch):
        T    = b['T']
        tlen = b['input_ids'].shape[0]
        face_batch[i, :T]  = b['face']
        lhand_batch[i, :T] = b['lhand']
        rhand_batch[i, :T] = b['rhand']
        body_batch[i, :T]  = b['body']
        lengths[i] = T
        label_batch[i, :tlen] = b['input_ids']
        decoder_input_batch[i, 1:tlen] = b['input_ids'][:tlen-1]

    return (face_batch, lhand_batch, rhand_batch, body_batch,
            lengths, label_batch, decoder_input_batch)


# ==========================================
# Model (same as Stage 2)
# ==========================================
def load_shubert(checkpoint_path):
    cfg = SHubertConfig()
    cfg.channels_embed_dim = 50
    cfg.channels_hand_embed_dim = 42
    cfg.channels_pose_embed_dim = 14
    cfg.encoder_embed_dim = 768

    model = SHubertModel(cfg)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)

    lora_config = LoraConfig(
        r=1, lora_alpha=1,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.0, bias="none",
    )
    model = get_peft_model(model, lora_config)
    return model


class SLTModel(nn.Module):
    def __init__(self, shubert, byt5, shubert_dim=768, byt5_dim=1536):
        super().__init__()
        self.shubert = shubert
        self.byt5    = byt5
        self.proj    = nn.Linear(shubert_dim, byt5_dim)
        self.layer_weights = nn.Parameter(torch.ones(12))

    def encode_sign(self, face, lhand, rhand, body, lengths):
        B, max_T, _ = face.shape
        device = face.device
        all_feats = []

        for i in range(B):
            T = lengths[i].item()
            source = [{
                "face":               face[i, :T],
                "left_hand":          lhand[i, :T],
                "right_hand":         rhand[i, :T],
                "body_posture":       body[i, :T],
                "label_face":         torch.zeros(T, 1, device=device),
                "label_left_hand":    torch.zeros(T, 1, device=device),
                "label_right_hand":   torch.zeros(T, 1, device=device),
                "label_body_posture": torch.zeros(T, 1, device=device),
            }]
            result = self.shubert.extract_features(
                source, padding_mask=None, kmeans_labels=None, mask=False
            )
            layer_feats = torch.stack(
                [layer[-1].squeeze(1) for layer in result['layer_results']], dim=0
            )
            weights = torch.softmax(self.layer_weights, dim=0)
            feat = (layer_feats * weights.unsqueeze(-1).unsqueeze(-1)).sum(0)
            all_feats.append(feat)

        max_feat_T = max(f.shape[0] for f in all_feats)
        encoder_hidden = torch.zeros(B, max_feat_T, 768, device=device)
        attention_mask = torch.zeros(B, max_feat_T, dtype=torch.long, device=device)
        for i, feat in enumerate(all_feats):
            T = feat.shape[0]
            encoder_hidden[i, :T] = feat
            attention_mask[i, :T] = 1

        encoder_hidden = self.proj(encoder_hidden)
        return encoder_hidden, attention_mask

    def forward(self, face, lhand, rhand, body, lengths, labels, decoder_input_ids):
        encoder_hidden, attention_mask = self.encode_sign(face, lhand, rhand, body, lengths)
        outputs = self.byt5(
            inputs_embeds=encoder_hidden,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        return outputs.loss


# ==========================================
# Main
# ==========================================
def main(args):
    wandb.init(
        project="slt_stage1_youtube",
        name=os.path.basename(args.output_dir),
        config=vars(args)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)

    print('Loading ByT5...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.byt5_path)
    byt5      = AutoModelForSeq2SeqLM.from_pretrained(args.byt5_path)

    print(f'Loading SHuBERT from {args.checkpoint_path}...', flush=True)
    shubert = load_shubert(args.checkpoint_path)
    model   = SLTModel(shubert, byt5).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params:,}, Trainable: {train_params:,}', flush=True)

    dataset = YouTubeASLSLTDataset(
        args.features_base,
        args.json_path,
        tokenizer,
        args.max_frames,
        args.max_text_len,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=2, collate_fn=collate_fn)

    shubert_params = [p for n, p in model.named_parameters() if 'shubert' in n and p.requires_grad]
    other_params   = [p for n, p in model.named_parameters() if 'shubert' not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': shubert_params, 'lr': args.lr_shubert},
        {'params': other_params,   'lr': args.lr},
    ], weight_decay=0.1)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=1e-6
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # 续训
    global_step = 0
    last_ckpt = os.path.join(args.output_dir, 'last_checkpoint.pt')
    if os.path.exists(last_ckpt):
        print(f'Resuming from {last_ckpt}...', flush=True)
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        global_step = ckpt.get('global_step', 0)
        print(f'Resumed from step {global_step}', flush=True)

    print(f'Starting Stage 1 training for {args.max_steps} steps...', flush=True)
    model.train()

    while global_step < args.max_steps:
        for batch in loader:
            if global_step >= args.max_steps:
                break

            face, lhand, rhand, body, lengths, labels, decoder_input_ids = batch
            face              = face.to(device)
            lhand             = lhand.to(device)
            rhand             = rhand.to(device)
            body              = body.to(device)
            lengths           = lengths.to(device)
            labels            = labels.to(device)
            decoder_input_ids = decoder_input_ids.to(device)

            optimizer.zero_grad()
            loss = model(face, lhand, rhand, body, lengths, labels, decoder_input_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1

            if global_step % 100 == 0:
                lr_shubert = optimizer.param_groups[0]['lr']
                lr_other   = optimizer.param_groups[1]['lr']
                print(f'Step {global_step}/{args.max_steps} | Loss: {loss.item():.4f} | '
                      f'lr_shubert: {lr_shubert:.2e} | lr_other: {lr_other:.2e}', flush=True)
                wandb.log({
                    'train/loss':       loss.item(),
                    'lr/shubert':       lr_shubert,
                    'lr/other':         lr_other,
                    'step':             global_step,
                })

            # 每 5000 步保存一次
            if global_step % 5000 == 0:
                torch.save({
                    'global_step':          global_step,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(args.output_dir, f'checkpoint_step{global_step}.pt'))
                print(f'Checkpoint saved at step {global_step}', flush=True)

            if global_step % 500 == 0:
                torch.save({
                    'global_step':          global_step,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(args.output_dir, 'last_checkpoint.pt'))

    # 保存最终模型
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, 'stage1_final_model.pt'))
    print('Stage 1 training complete!', flush=True)
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_base',   type=str,
                        default='/scratch/project_465002480/YouTube_ASL/features')
    parser.add_argument('--json_path',       type=str,
                        default='/scratch/project_465002480/YouTube_ASL/YT.translations.train.json')
    parser.add_argument('--checkpoint_path', type=str,
                        default='/scratch/project_465002480/YouTube_ASL/training_logs/run_full_w8_200k/ckpt/checkpoint160.pt')
    parser.add_argument('--byt5_path',       type=str,
                        default='/scratch/project_465003161/byt5-base')
    parser.add_argument('--output_dir',      type=str,
                        default='/scratch/project_465003161/slt_stage1')
    parser.add_argument('--batch_size',   type=int,   default=4)
    parser.add_argument('--max_steps',    type=int,   default=250000)
    parser.add_argument('--lr',           type=float, default=5e-4)
    parser.add_argument('--lr_shubert',   type=float, default=5e-5)
    parser.add_argument('--max_frames',   type=int,   default=500)
    parser.add_argument('--max_text_len', type=int,   default=256)
    args = parser.parse_args()
    main(args)
