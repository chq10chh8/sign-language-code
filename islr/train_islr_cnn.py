"""
Simple 1D CNN Baseline for Isolated Sign Language Recognition
- 输入：face/lhand/rhand/body 四个流拼接 [T, 148]
- 模型：三层 1D CNN + Global Average Pooling + 分类头
- 用途：作为基准对比 SHuBERT 和 Hi-TRS
"""

import numpy as np
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
import wandb


# ==========================================
# Dataset
# ==========================================
class WLASLDataset(Dataset):
    def __init__(self, features_base, meta, split, label2idx, max_frames=500):
        self.samples = []
        self.features_base = Path(features_base)
        self.max_frames = max_frames

        for idx, item in enumerate(meta):
            if item['split'] != split:
                continue
            gloss = item['gloss']
            if gloss not in label2idx:
                continue

            face_path  = self.features_base / 'face'  / f'wlasl_{idx}_face.npy'
            lhand_path = self.features_base / 'lhand' / f'wlasl_{idx}_left_hand.npy'
            rhand_path = self.features_base / 'rhand' / f'wlasl_{idx}_right_hand.npy'
            body_path  = self.features_base / 'body'  / f'wlasl_{idx}_body.npy'

            if not all([p.exists() for p in [face_path, lhand_path, rhand_path, body_path]]):
                continue

            self.samples.append((
                str(face_path), str(lhand_path),
                str(rhand_path), str(body_path),
                label2idx[gloss]
            ))

        print(f'{split}: {len(self.samples)} samples')

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

        # 把 -9999 的无效帧置零
        face[np.any(face == -9999, axis=1)]  = 0
        lhand[np.any(lhand == -9999, axis=1)] = 0
        rhand[np.any(rhand == -9999, axis=1)] = 0
        body[np.any(body == -9999, axis=1)]   = 0

        # 拼接四个流 [T, 148]
        feat = np.concatenate([face, lhand, rhand, body], axis=1)

        return {
            'feat':  torch.FloatTensor(feat),
            'label': label,
            'T': T,
        }


def collate_fn(batch):
    max_T = max(b['T'] for b in batch)
    B = len(batch)
    feat_batch = torch.zeros(B, max_T, 148)
    labels  = torch.zeros(B, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        T = b['T']
        feat_batch[i, :T] = b['feat']
        labels[i]  = b['label']
        lengths[i] = T

    return feat_batch, labels, lengths


# ==========================================
# CNN Model
# ==========================================
class CNN1DBaseline(nn.Module):
    """
    简单的 1D CNN 基准模型
    输入：[B, T, 148]
    输出：[B, num_classes]
    """
    def __init__(self, input_dim=148, num_classes=2000, dropout=0.3):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Layer 2
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Layer 3
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        total = sum(p.numel() for p in self.parameters())
        print(f'CNN1DBaseline total parameters: {total:,}')

    def forward(self, x, lengths=None):
        # x: [B, T, 148] → [B, 148, T] for Conv1d
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        # Global Average Pooling: [B, 512, T'] → [B, 512]
        x = x.mean(dim=2)
        return self.classifier(x)


# ==========================================
# 训练 & 评估
# ==========================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for feat, labels, lengths in loader:
        feat    = feat.to(device)
        labels  = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(feat, lengths)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, device, topk=(1, 5, 10)):
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for feat, labels, lengths in loader:
            feat    = feat.to(device)
            lengths = lengths.to(device)
            all_logits.append(model(feat, lengths).cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, 0)
    all_labels = torch.cat(all_labels, 0)

    results = {}
    for k in topk:
        preds = all_logits.topk(k, dim=1).indices
        results[f'Rec@{k}'] = preds.eq(all_labels.unsqueeze(1)).any(1).float().mean().item()
    return results


# ==========================================
# Main
# ==========================================
def main(args):
    wandb.init(
        project="islr_cnn_baseline",
        name=os.path.basename(args.output_dir),
        config=vars(args)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    with open(args.meta_path) as f:
        meta = json.load(f)

    all_glosses = sorted(set(item['gloss'] for item in meta))
    label2idx   = {g: i for i, g in enumerate(all_glosses)}
    num_classes  = len(all_glosses)
    print(f'Total classes: {num_classes}')

    train_dataset = WLASLDataset(args.features_base, meta, 'train', label2idx, args.max_frames)
    val_dataset   = WLASLDataset(args.features_base, meta, 'val',   label2idx, args.max_frames)
    test_dataset  = WLASLDataset(args.features_base, meta, 'test',  label2idx, args.max_frames)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, collate_fn=collate_fn,
                              prefetch_factor=4, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, collate_fn=collate_fn,
                              prefetch_factor=4, persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size,
                              shuffle=False, num_workers=4, collate_fn=collate_fn,
                              prefetch_factor=4, persistent_workers=True)

    model = CNN1DBaseline(
        input_dim=148,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    os.makedirs(args.output_dir, exist_ok=True)

    best_val_rec1  = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_results = eval_epoch(model, val_loader, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(f'Epoch {epoch+1}/{args.epochs} | '
              f'Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Rec@1: {val_results["Rec@1"]:.4f} | '
              f'Val Rec@5: {val_results["Rec@5"]:.4f} | LR: {lr:.2e}')

        wandb.log({
            'epoch': epoch + 1,
            'train/loss': train_loss,
            'train/acc':  train_acc,
            'val/Rec@1':  val_results['Rec@1'],
            'val/Rec@5':  val_results['Rec@5'],
            'val/Rec@10': val_results['Rec@10'],
            'lr': lr,
        })

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_rec1': best_val_rec1,
        }, os.path.join(args.output_dir, 'last_checkpoint.pt'))

        if val_results['Rec@1'] > best_val_rec1:
            best_val_rec1 = val_results['Rec@1']
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, 'best_model.pt'))
            print(f'  Best model saved! Val Rec@1: {best_val_rec1:.4f}')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    print('\nLoading best model for test evaluation...')
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    test_results = eval_epoch(model, test_loader, device)

    print('=' * 50)
    print('Test Results:')
    for k, v in test_results.items():
        print(f'  {k}: {v:.4f}')
    print('=' * 50)

    with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
        for k, v in test_results.items():
            f.write(f'{k}: {v:.4f}\n')

    wandb.log({f'test/{k}': v for k, v in test_results.items()})
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_base', type=str,
                        default='/scratch/project_465002480/YouTube_ASL/wlasl/features')
    parser.add_argument('--meta_path', type=str,
                        default='/scratch/project_465002480/YouTube_ASL/wlasl/WLASL_parsed_data.json')
    parser.add_argument('--output_dir', type=str,
                        default='/scratch/project_465002480/YouTube_ASL/wlasl/islr_cnn_results')
    parser.add_argument('--batch_size',   type=int,   default=64)
    parser.add_argument('--epochs',       type=int,   default=125)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout',      type=float, default=0.3)
    parser.add_argument('--max_frames',   type=int,   default=500)
    parser.add_argument('--patience',     type=int,   default=20)
    args = parser.parse_args()
    main(args)
