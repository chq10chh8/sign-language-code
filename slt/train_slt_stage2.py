"""
Sign Language Translation (SLT) - Stage 2
Fine-tuning on How2Sign starting from Stage 1 model
Stage 1 model = SHuBERT + projection + ByT5 trained on YouTube-ASL
"""

import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
import pandas as pd
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
class How2SignDataset(Dataset):
    def __init__(self, features_base, csv_path, tokenizer,
                 max_frames=500, max_text_len=256, split='train'):
        self.features_base = Path(features_base)
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.max_text_len = max_text_len
        self.samples = []

        df = pd.read_csv(csv_path, sep='\t')
        for _, row in df.iterrows():
            name = row['SENTENCE_NAME']
            text = str(row['SENTENCE']).strip()
            if not text or text == 'nan':
                continue

            face_path  = self.features_base / split / 'face'  / f'{name}_face.npy'
            lhand_path = self.features_base / split / 'lhand' / f'{name}_left_hand.npy'
            rhand_path = self.features_base / split / 'rhand' / f'{name}_right_hand.npy'
            body_path  = self.features_base / split / 'body'  / f'{name}_body.npy'

            if not all([p.exists() for p in [face_path, lhand_path, rhand_path, body_path]]):
                continue

            self.samples.append((
                str(face_path), str(lhand_path),
                str(rhand_path), str(body_path),
                text
            ))

        print(f'{split}: {len(self.samples)} samples', flush=True)

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
# Model
# ==========================================
def build_slt_model(byt5_path, shubert_dim=768, byt5_dim=1536):
    """构建空的 SLT 模型结构（用于加载 Stage 1 权重）"""
    cfg = SHubertConfig()
    cfg.channels_embed_dim = 50
    cfg.channels_hand_embed_dim = 42
    cfg.channels_pose_embed_dim = 14
    cfg.encoder_embed_dim = 768

    shubert = SHubertModel(cfg)
    lora_config = LoraConfig(
        r=1, lora_alpha=1,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.0, bias="none",
    )
    shubert = get_peft_model(shubert, lora_config)

    byt5 = AutoModelForSeq2SeqLM.from_pretrained(byt5_path)
    model = SLTModel(shubert, byt5, shubert_dim, byt5_dim)
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

    def generate(self, face, lhand, rhand, body, lengths, tokenizer,
                 max_length=256, num_beams=5):
        encoder_hidden, attention_mask = self.encode_sign(face, lhand, rhand, body, lengths)
        generated = self.byt5.generate(
            inputs_embeds=encoder_hidden,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        return tokenizer.batch_decode(generated, skip_special_tokens=True)


# ==========================================
# BLEU Evaluation
# ==========================================
def evaluate_bleu(model, loader, tokenizer, device):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    model.eval()
    hypotheses, references = [], []

    with torch.no_grad():
        for batch in loader:
            face, lhand, rhand, body, lengths, labels, decoder_input_ids = batch
            face    = face.to(device)
            lhand   = lhand.to(device)
            rhand   = rhand.to(device)
            body    = body.to(device)
            lengths = lengths.to(device)

            preds = model.generate(face, lhand, rhand, body, lengths, tokenizer)

            for pred, label in zip(preds, labels):
                label_ids = label[label != -100]
                ref_text  = tokenizer.decode(label_ids, skip_special_tokens=True)
                hypotheses.append(pred.split())
                references.append([ref_text.split()])

    sf    = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, hypotheses, weights=(1,0,0,0), smoothing_function=sf) * 100
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5,0.5,0,0), smoothing_function=sf) * 100
    bleu4 = corpus_bleu(references, hypotheses, smoothing_function=sf) * 100
    return bleu1, bleu2, bleu4


# ==========================================
# Main
# ==========================================
def main(args):
    wandb.init(
        project="slt_how2sign_stage2",
        name=os.path.basename(args.output_dir),
        config=vars(args)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)

    print('Loading tokenizer...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.byt5_path)

    print('Building model...', flush=True)
    model = build_slt_model(args.byt5_path)

    print(f'Loading Stage 1 model from {args.stage1_model_path}...', flush=True)
    model.load_state_dict(torch.load(args.stage1_model_path, map_location='cpu'))
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params:,}, Trainable: {train_params:,}', flush=True)

    meta_base = Path('/scratch/project_465003161/How2Sign_Holistic/how2sign_holistic_features/metadata')

    train_dataset = How2SignDataset(
        args.features_base, meta_base / 'how2sign_realigned_train.csv',
        tokenizer, args.max_frames, args.max_text_len, split='train'
    )
    val_dataset = How2SignDataset(
        args.features_base, meta_base / 'how2sign_realigned_val.csv',
        tokenizer, args.max_frames, args.max_text_len, split='val'
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, collate_fn=collate_fn)

    shubert_params = [p for n, p in model.named_parameters() if 'shubert' in n and p.requires_grad]
    other_params   = [p for n, p in model.named_parameters() if 'shubert' not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': shubert_params, 'lr': args.lr_shubert},
        {'params': other_params,   'lr': args.lr},
    ], weight_decay=0.1)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # 续训
    start_epoch = 0
    best_bleu   = 0
    last_ckpt   = os.path.join(args.output_dir, 'last_checkpoint.pt')
    if os.path.exists(last_ckpt):
        print(f'Resuming from {last_ckpt}...', flush=True)
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_bleu   = ckpt.get('best_bleu', 0)
        print(f'Resumed from epoch {start_epoch}', flush=True)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss  = 0
        global_step = 0

        for batch_idx, batch in enumerate(train_loader):
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

            total_loss  += loss.item()
            global_step += 1

            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f}', flush=True)
                wandb.log({
                    'train/loss_per_batch': loss.item(),
                    'lr/shubert': optimizer.param_groups[0]['lr'],
                    'lr/other':   optimizer.param_groups[1]['lr'],
                })

        scheduler.step()
        avg_loss   = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        # BLEU 每5个epoch算一次
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            bleu1, bleu2, bleu4 = evaluate_bleu(model, val_loader, tokenizer, device)
            print(f'Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | '
                  f'BLEU-1: {bleu1:.2f} | BLEU-2: {bleu2:.2f} | BLEU-4: {bleu4:.2f} | '
                  f'Time: {epoch_time:.0f}s', flush=True)
            wandb.log({
                'epoch':            epoch + 1,
                'train/loss':       avg_loss,
                'train/epoch_time': epoch_time,
                'val/BLEU_1':       bleu1,
                'val/BLEU_2':       bleu2,
                'val/BLEU_4':       bleu4,
            })
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch{epoch+1}.pt"))
            if bleu4 > best_bleu:
                best_bleu = bleu4
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, 'best_model.pt'))
                print(f'  Best model saved! BLEU-4: {best_bleu:.2f}', flush=True)
        else:
            print(f'Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | '
                  f'Time: {epoch_time:.0f}s', flush=True)
            wandb.log({
                'epoch':            epoch + 1,
                'train/loss':       avg_loss,
                'train/epoch_time': epoch_time,
            })

        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_bleu':            best_bleu,
        }, os.path.join(args.output_dir, 'last_checkpoint.pt'))

    # Test evaluation
    print('\nEvaluating on test set...', flush=True)
    test_dataset = How2SignDataset(
        args.features_base, meta_base / 'how2sign_realigned_test.csv',
        tokenizer, args.max_frames, args.max_text_len, split='test'
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, collate_fn=collate_fn)

    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    test_bleu1, test_bleu2, test_bleu4 = evaluate_bleu(model, test_loader, tokenizer, device)

    print(f'Test BLEU-1: {test_bleu1:.2f}')
    print(f'Test BLEU-2: {test_bleu2:.2f}')
    print(f'Test BLEU-4: {test_bleu4:.2f}')

    with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
        f.write(f'BLEU-1: {test_bleu1:.2f}\n')
        f.write(f'BLEU-2: {test_bleu2:.2f}\n')
        f.write(f'BLEU-4: {test_bleu4:.2f}\n')

    wandb.log({
        'test/BLEU_1': test_bleu1,
        'test/BLEU_2': test_bleu2,
        'test/BLEU_4': test_bleu4,
    })
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_base',    type=str,
                        default='/scratch/project_465003161/How2Sign_features')
    parser.add_argument('--stage1_model_path', type=str,
                        default='/scratch/project_465003161/slt_stage1/stage1_final_model.pt')
    parser.add_argument('--byt5_path',        type=str,
                        default='/scratch/project_465003161/byt5-base')
    parser.add_argument('--output_dir',       type=str,
                        default='/scratch/project_465003161/slt_results/stage1_then_stage2')
    parser.add_argument('--batch_size',   type=int,   default=4)
    parser.add_argument('--epochs',       type=int,   default=10)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--lr_shubert',   type=float, default=1e-5)
    parser.add_argument('--max_frames',   type=int,   default=500)
    parser.add_argument('--max_text_len', type=int,   default=256)
    args = parser.parse_args()
    main(args)
