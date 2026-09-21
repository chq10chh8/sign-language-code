"""
SLT 评估脚本
- 计算 BLEU-1/2/4
- 计算 BLEURT
- 生成翻译样例
"""

import numpy as np
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.insert(0, '/scratch/project_465002480/YouTube_ASL/custom_python_libs')
sys.path.insert(0, '/scratch/project_465002480/SHuBERT/fairseq')
sys.path.insert(0, '/scratch/project_465002480/SHuBERT/fairseq/examples')

from examples.shubert.models.shubert import SHubertModel, SHubertConfig
from peft import LoraConfig, get_peft_model


# ==========================================
# Dataset & Model (same as train_slt.py)
# ==========================================
class How2SignDataset(Dataset):
    def __init__(self, features_base, csv_path, tokenizer,
                 max_frames=500, max_text_len=256, split='test'):
        self.features_base = Path(features_base)
        self.tokenizer = tokenizer
        self.max_frames = max_frames
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
            max_length=256,
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
            'text':      text,
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
    texts       = [b['text'] for b in batch]

    for i, b in enumerate(batch):
        T    = b['T']
        tlen = b['input_ids'].shape[0]
        face_batch[i, :T]  = b['face']
        lhand_batch[i, :T] = b['lhand']
        rhand_batch[i, :T] = b['rhand']
        body_batch[i, :T]  = b['body']
        lengths[i] = T
        label_batch[i, :tlen] = b['input_ids']

    return face_batch, lhand_batch, rhand_batch, body_batch, lengths, label_batch, texts


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
# Main
# ==========================================
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)

    print('Loading ByT5...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.byt5_path)
    byt5      = AutoModelForSeq2SeqLM.from_pretrained(args.byt5_path)

    print(f'Loading SHuBERT...', flush=True)
    shubert = load_shubert(args.shubert_checkpoint)
    model   = SLTModel(shubert, byt5).to(device)

    print(f'Loading fine-tuned model from {args.model_path}...', flush=True)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    meta_base = Path('/scratch/project_465003161/How2Sign_Holistic/how2sign_holistic_features/metadata')

    dataset = How2SignDataset(
        args.features_base,
        meta_base / f'how2sign_realigned_{args.split}.csv',
        tokenizer, split=args.split
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=2, collate_fn=collate_fn)

    # 生成所有翻译
    all_preds = []
    all_refs  = []

    print('Generating translations...', flush=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            face, lhand, rhand, body, lengths, labels, texts = batch
            face    = face.to(device)
            lhand   = lhand.to(device)
            rhand   = rhand.to(device)
            body    = body.to(device)
            lengths = lengths.to(device)

            preds = model.generate(face, lhand, rhand, body, lengths, tokenizer)
            all_preds.extend(preds)
            all_refs.extend(texts)

            if (batch_idx + 1) % 50 == 0:
                print(f'  {batch_idx+1}/{len(loader)} batches done', flush=True)

    print(f'Total samples: {len(all_preds)}', flush=True)

    # ==========================================
    # BLEU
    # ==========================================
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    sf = SmoothingFunction().method1
    hyps = [p.split() for p in all_preds]
    refs = [[r.split()] for r in all_refs]

    bleu1 = corpus_bleu(refs, hyps, weights=(1,0,0,0), smoothing_function=sf) * 100
    bleu2 = corpus_bleu(refs, hyps, weights=(0.5,0.5,0,0), smoothing_function=sf) * 100
    bleu4 = corpus_bleu(refs, hyps, smoothing_function=sf) * 100

    print(f'\n=== BLEU Results ({args.split}) ===')
    print(f'BLEU-1: {bleu1:.2f}')
    print(f'BLEU-2: {bleu2:.2f}')
    print(f'BLEU-4: {bleu4:.2f}')

    # ==========================================
    # BLEURT
    # ==========================================
    print('\nComputing BLEURT...', flush=True)
    try:
        from evaluate import load
        bleurt = load('bleurt', 'bleurt-20')
        bleurt_results = bleurt.compute(
            predictions=all_preds,
            references=all_refs
        )
        bleurt_score = sum(bleurt_results['scores']) / len(bleurt_results['scores']) * 100
        print(f'BLEURT: {bleurt_score:.2f}')
    except Exception as e:
        print(f'BLEURT failed: {e}')
        bleurt_score = None

    # ==========================================
    # 翻译样例
    # ==========================================
    print(f'\n=== Translation Examples ({args.split}) ===')
    for i in range(min(10, len(all_preds))):
        print(f'\n[{i+1}]')
        print(f'  Reference:  {all_refs[i]}')
        print(f'  Prediction: {all_preds[i]}')

    # ==========================================
    # 保存结果
    # ==========================================
    model_name = Path(args.model_path).stem
    out_dir = Path(args.model_path).parent
    result_path = out_dir / f'eval_{args.split}_{model_name}_results.txt'
    with open(result_path, 'w') as f:
        f.write(f'Split: {args.split}\n')
        f.write(f'Model: {args.model_path}\n')
        f.write(f'Samples: {len(all_preds)}\n\n')
        f.write(f'BLEU-1: {bleu1:.2f}\n')
        f.write(f'BLEU-2: {bleu2:.2f}\n')
        f.write(f'BLEU-4: {bleu4:.2f}\n')
        if bleurt_score is not None:
            f.write(f'BLEURT: {bleurt_score:.2f}\n')
        f.write('\n=== Examples ===\n')
        for i in range(min(20, len(all_preds))):
            f.write(f'\n[{i+1}]\n')
            f.write(f'  Ref:  {all_refs[i]}\n')
            f.write(f'  Pred: {all_preds[i]}\n')

    print(f'\nResults saved to {result_path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_base',    type=str,
                        default='/scratch/project_465003161/How2Sign_features')
    parser.add_argument('--shubert_checkpoint', type=str,
                        default='/scratch/project_465002480/YouTube_ASL/training_logs/run_full_w8_200k/ckpt/checkpoint160.pt')
    parser.add_argument('--model_path',       type=str,
                        default='/scratch/project_465003161/slt_results/ckpt160/best_model.pt')
    parser.add_argument('--byt5_path',        type=str,
                        default='/scratch/project_465003161/byt5-base')
    parser.add_argument('--split',            type=str,   default='test',
                        choices=['val', 'test'])
    parser.add_argument('--batch_size',       type=int,   default=4)
    parser.add_argument('--max_frames',       type=int,   default=500)
    args = parser.parse_args()
    main(args)
