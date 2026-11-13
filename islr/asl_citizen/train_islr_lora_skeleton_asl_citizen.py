"""
ISLR LoRA 微调脚本 - 骨架特征版本
- 输入: face(50), lhand(42), rhand(42), body(14)
- 加载官方 SHuBERT 预训练权重 (checkpoint_836_400000.pt)
- 手动实现 LoRA rank-1 微调 (不依赖 peft)
- ASL Citizen 数据集，特征来自 ASL_Citizen_skeleton/
"""

import numpy as np
import pandas as pd
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
import wandb

sys.path.insert(0, '/scratch/project_465003161/YouTube_ASL/custom_python_libs')
sys.path.insert(0, '/scratch/project_465003161/SHuBERT/fairseq')
sys.path.insert(0, '/scratch/project_465003161/SHuBERT/fairseq/examples')

from examples.shubert.models.shubert import SHubertModel, SHubertConfig


# ==========================================
# 手动 LoRA 实现
# ==========================================
class LoRALinear(nn.Module):
    def __init__(self, linear, r=1, alpha=1):
        super().__init__()
        self.linear = linear
        self.r = r
        self.scale = alpha / r
        d_in = linear.weight.shape[1]
        d_out = linear.weight.shape[0]
        self.lora_A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)
        self.bias = linear.bias
        self.weight = linear.weight

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def apply_lora(model, target_names, r=1, alpha=1):
    """递归替换目标层为 LoRALinear"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in target_names:
            setattr(model, name, LoRALinear(module, r=r, alpha=alpha))
        else:
            apply_lora(module, target_names, r=r, alpha=alpha)


# ==========================================
# Dataset
# ==========================================
class ASLCitizenSkeletonDataset(Dataset):
    def __init__(self, skeleton_base, csv_path, label2idx, max_frames=500):
        self.samples = []
        self.skeleton_base = Path(skeleton_base)
        self.max_frames = max_frames

        df = pd.read_csv(csv_path)
        split = Path(csv_path).stem  # train / val / test

        for _, row in df.iterrows():
            gloss = row['Gloss']
            if gloss not in label2idx:
                continue

            stem = Path(row['Video file']).stem

            face_path  = self.skeleton_base / split / 'face'  / f'{stem}_face.npy'
            lhand_path = self.skeleton_base / split / 'lhand' / f'{stem}_left_hand.npy'
            rhand_path = self.skeleton_base / split / 'rhand' / f'{stem}_right_hand.npy'
            body_path  = self.skeleton_base / split / 'body'  / f'{stem}_body.npy'

            if not all(p.exists() for p in [face_path, lhand_path, rhand_path, body_path]):
                continue

            self.samples.append((
                str(face_path), str(lhand_path),
                str(rhand_path), str(body_path),
                label2idx[gloss]
            ))

        print(f'{split}: {len(self.samples)} samples', flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        face_path, lhand_path, rhand_path, body_path, label = self.samples[idx]

        face  = np.load(face_path).astype(np.float32)
        lhand = np.load(lhand_path).astype(np.float32)
        rhand = np.load(rhand_path).astype(np.float32)
        body  = np.load(body_path).astype(np.float32)

        T = min(face.shape[0], self.max_frames)
        face, lhand, rhand, body = face[:T], lhand[:T], rhand[:T], body[:T]

        return {
            'face':  torch.FloatTensor(face),
            'lhand': torch.FloatTensor(lhand),
            'rhand': torch.FloatTensor(rhand),
            'body':  torch.FloatTensor(body),
            'label': label,
            'T': T
        }


def collate_fn(batch):
    max_T = max(b['T'] for b in batch)
    B = len(batch)

    # 骨架特征维度: face=50, lhand=42, rhand=42, body=14
    face_batch  = torch.zeros(B, max_T, 50)
    lhand_batch = torch.zeros(B, max_T, 42)
    rhand_batch = torch.zeros(B, max_T, 42)
    body_batch  = torch.zeros(B, max_T, 14)
    labels      = torch.zeros(B, dtype=torch.long)
    lengths     = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        T = b['T']
        face_batch[i, :T]  = b['face']
        lhand_batch[i, :T] = b['lhand']
        rhand_batch[i, :T] = b['rhand']
        body_batch[i, :T]  = b['body']
        labels[i]  = b['label']
        lengths[i] = T

    return face_batch, lhand_batch, rhand_batch, body_batch, labels, lengths


# ==========================================
# Model
# ==========================================
def load_shubert_with_lora(checkpoint_path, r=1, alpha=1):
    cfg = SHubertConfig()
    # 骨架特征维度
    cfg.channels_embed_dim      = 50   # face
    cfg.channels_hand_embed_dim = 42   # lhand / rhand
    cfg.channels_pose_embed_dim = 14   # body

    model = SHubertModel(cfg)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt

    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_state and v.shape == model_state[k].shape}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f'Missing keys: {len(missing)}, Unexpected: {len(unexpected)}', flush=True)

    for param in model.parameters():
        param.requires_grad_(False)

    target_names = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}
    apply_lora(model, target_names, r=r, alpha=alpha)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f'Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)', flush=True)

    return model


class ISLRModel(nn.Module):
    def __init__(self, shubert, num_classes):
        super().__init__()
        self.shubert = shubert
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.Linear(768, num_classes)
        )

    def forward(self, face, lhand, rhand, body, lengths):
        B = face.shape[0]
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
            feat = result['x'].mean(dim=1)  # [1, 768]
            all_feats.append(feat.squeeze(0))

        feats = torch.stack(all_feats)  # [B, 768]
        return self.classifier(feats)


# ==========================================
# Eval helper
# ==========================================
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for face, lhand, rhand, body, labels, lengths in loader:
            face    = face.to(device)
            lhand   = lhand.to(device)
            rhand   = rhand.to(device)
            body    = body.to(device)
            lengths = lengths.to(device)
            logits  = model(face, lhand, rhand, body, lengths)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    rec1  = all_logits.topk(1,  dim=1).indices.eq(all_labels.unsqueeze(1)).any(1).float().mean().item()
    rec5  = all_logits.topk(5,  dim=1).indices.eq(all_labels.unsqueeze(1)).any(1).float().mean().item()
    rec10 = all_logits.topk(10, dim=1).indices.eq(all_labels.unsqueeze(1)).any(1).float().mean().item()
    return rec1, rec5, rec10


# ==========================================
# Main
# ==========================================
def main(args):
    wandb.init(
        project="islr_skeleton_lora",
        name=f"asl_citizen_skeleton_lora_r{args.lora_r}",
        config=vars(args)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)

    # 从训练集构建 label2idx（只用有效 gloss）
    train_df = pd.read_csv(args.train_csv)
    skeleton_base = Path(args.skeleton_base)

    valid_glosses = set()
    for _, row in train_df.iterrows():
        stem = Path(row['Video file']).stem
        face_path = skeleton_base / 'train' / 'face' / f'{stem}_face.npy'
        if face_path.exists():
            valid_glosses.add(row['Gloss'])

    label2idx   = {g: i for i, g in enumerate(sorted(valid_glosses))}
    num_classes = len(label2idx)
    print(f'Classes: {num_classes}', flush=True)

    train_dataset = ASLCitizenSkeletonDataset(args.skeleton_base, args.train_csv, label2idx, args.max_frames)
    val_dataset   = ASLCitizenSkeletonDataset(args.skeleton_base, args.val_csv,   label2idx, args.max_frames)
    test_dataset  = ASLCitizenSkeletonDataset(args.skeleton_base, args.test_csv,  label2idx, args.max_frames)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, collate_fn=collate_fn,
                              drop_last=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, collate_fn=collate_fn,
                              persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size,
                              shuffle=False, num_workers=2, collate_fn=collate_fn,
                              persistent_workers=True)

    print(f'Loading SHuBERT from {args.checkpoint_path}...', flush=True)
    shubert = load_shubert_with_lora(args.checkpoint_path, r=args.lora_r, alpha=args.lora_r)
    model   = ISLRModel(shubert, num_classes).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_rec1 = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_idx, (face, lhand, rhand, body, labels, lengths) in enumerate(train_loader):
            face    = face.to(device)
            lhand   = lhand.to(device)
            rhand   = rhand.to(device)
            body    = body.to(device)
            labels  = labels.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            logits = model(face, lhand, rhand, body, lengths)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}', flush=True)

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        rec1, rec5, rec10 = evaluate(model, val_loader, device)

        print(f'Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | '
              f'Val Rec@1: {rec1:.4f} | Rec@5: {rec5:.4f} | Rec@10: {rec10:.4f}', flush=True)

        wandb.log({
            'epoch':      epoch + 1,
            'train/loss': avg_loss,
            'val/Rec@1':  rec1,
            'val/Rec@5':  rec5,
            'val/Rec@10': rec10,
        })

        if rec1 > best_val_rec1:
            best_val_rec1 = rec1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
            print(f'  Best model saved! Val Rec@1: {best_val_rec1:.4f}', flush=True)
        else:
            patience_counter += 1
            print(f'  No improvement. Patience: {patience_counter}/{args.patience}', flush=True)
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch+1}', flush=True)
                break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_rec1': best_val_rec1,
        }, os.path.join(args.output_dir, 'last_checkpoint.pt'))

    print('\nEvaluating on test set...', flush=True)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    rec1, rec5, rec10 = evaluate(model, test_loader, device)

    print(f'Test Rec@1: {rec1:.4f} | Rec@5: {rec5:.4f} | Rec@10: {rec10:.4f}')

    with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
        f.write(f'Rec@1:  {rec1:.4f}\nRec@5:  {rec5:.4f}\nRec@10: {rec10:.4f}\n')

    wandb.log({'test/Rec@1': rec1, 'test/Rec@5': rec5, 'test/Rec@10': rec10})
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skeleton_base',  type=str,
                        default='/scratch/project_465003161/ASL_Citizen_skeleton')
    parser.add_argument('--train_csv',      type=str,
                        default='/scratch/project_465003161/ASL_Citizen/ASL_Citizen/splits/train.csv')
    parser.add_argument('--val_csv',        type=str,
                        default='/scratch/project_465003161/ASL_Citizen/ASL_Citizen/splits/val.csv')
    parser.add_argument('--test_csv',       type=str,
                        default='/scratch/project_465003161/ASL_Citizen/ASL_Citizen/splits/test.csv')
    parser.add_argument('--checkpoint_path', type=str,
                        default='/scratch/project_465003161/shubert_official_weights/SHuBERT_ckpts/checkpoint_836_400000.pt')
    parser.add_argument('--output_dir',     type=str,
                        default='/scratch/project_465003161/asl_citizen_skeleton_islr_results')
    parser.add_argument('--batch_size',   type=int,   default=8)
    parser.add_argument('--epochs',       type=int,   default=125)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_frames',   type=int,   default=500)
    parser.add_argument('--lora_r',       type=int,   default=1)
    parser.add_argument('--patience',     type=int,   default=20)
    args = parser.parse_args()
    main(args)
