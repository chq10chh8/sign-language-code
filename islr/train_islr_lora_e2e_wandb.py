"""
端到端 LoRA ISLR 训练脚本
- 直接从 face/lhand/rhand/body npy 输入 SHuBERT
- LoRA rank-1 微调 SHuBERT
- BatchNorm + Linear 分类头
- 按论文设置: 125 epochs, Adam lr=1e-4, weight_decay=1e-4
"""

import numpy as np
import json
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
import wandb

sys.path.insert(0, '/scratch/project_465002480/YouTube_ASL/custom_python_libs')
sys.path.insert(0, '/scratch/project_465002480/SHuBERT/fairseq')
sys.path.insert(0, '/scratch/project_465002480/SHuBERT/fairseq/examples')

from examples.shubert.models.shubert import SHubertModel, SHubertConfig
from peft import LoraConfig, get_peft_model


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

            face_path = self.features_base / 'face' / f'wlasl_{idx}_face.npy'
            lhand_path = self.features_base / 'lhand' / f'wlasl_{idx}_left_hand.npy'
            rhand_path = self.features_base / 'rhand' / f'wlasl_{idx}_right_hand.npy'
            body_path = self.features_base / 'body' / f'wlasl_{idx}_body.npy'

            if not all([face_path.exists(), lhand_path.exists(),
                        rhand_path.exists(), body_path.exists()]):
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

        face = np.load(face_path).astype(np.float32)
        lhand = np.load(lhand_path).astype(np.float32)
        rhand = np.load(rhand_path).astype(np.float32)
        body = np.load(body_path).astype(np.float32)

        # 截断到 max_frames
        T = min(face.shape[0], self.max_frames)
        face = face[:T]
        lhand = lhand[:T]
        rhand = rhand[:T]
        body = body[:T]

        return {
            'face': torch.FloatTensor(face),
            'lhand': torch.FloatTensor(lhand),
            'rhand': torch.FloatTensor(rhand),
            'body': torch.FloatTensor(body),
            'label': label,
            'T': T
        }


def collate_fn(batch):
    """处理不同长度的序列，用padding对齐"""
    max_T = max(b['T'] for b in batch)
    B = len(batch)

    face_batch = torch.zeros(B, max_T, 50)
    lhand_batch = torch.zeros(B, max_T, 42)
    rhand_batch = torch.zeros(B, max_T, 42)
    body_batch = torch.zeros(B, max_T, 14)
    labels = torch.zeros(B, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        T = b['T']
        face_batch[i, :T] = b['face']
        lhand_batch[i, :T] = b['lhand']
        rhand_batch[i, :T] = b['rhand']
        body_batch[i, :T] = b['body']
        labels[i] = b['label']
        lengths[i] = T

    return face_batch, lhand_batch, rhand_batch, body_batch, labels, lengths


# ==========================================
# Model
# ==========================================
def load_shubert_with_lora(checkpoint_path):
    cfg = SHubertConfig()
    cfg.channels_embed_dim = 50
    cfg.channels_hand_embed_dim = 42
    cfg.channels_pose_embed_dim = 14
    cfg.encoder_embed_dim = 768

    model = SHubertModel(cfg)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    print("SHuBERT checkpoint loaded.")

    # LoRA rank-1
    lora_config = LoraConfig(
        r=1,
        lora_alpha=1,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


class ISLRLoRA(nn.Module):
    def __init__(self, shubert, num_classes, num_layers=12, hidden_dim=768):
        super().__init__()
        self.shubert = shubert
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, face, lhand, rhand, body, lengths):
        B, max_T, _ = face.shape
        device = face.device

        # 逐样本输入 SHuBERT（因为序列长度不同）
        all_feats = []
        for i in range(B):
            T = lengths[i].item()
            source = [{
                "face": face[i, :T],
                "left_hand": lhand[i, :T],
                "right_hand": rhand[i, :T],
                "body_posture": body[i, :T],
                "label_face": torch.zeros(T, 1, device=device),
                "label_left_hand": torch.zeros(T, 1, device=device),
                "label_right_hand": torch.zeros(T, 1, device=device),
                "label_body_posture": torch.zeros(T, 1, device=device),
            }]

            result = self.shubert.extract_features(
                source, padding_mask=None, kmeans_labels=None, mask=False
            )

            # 取所有层输出，对T帧取平均 → [12, 768]
            layer_feats = []
            for layer in result['layer_results']:
                layer_out = layer[-1].squeeze(1)  # [T, 768]
                layer_feats.append(layer_out.mean(0))  # [768]
            layer_feats = torch.stack(layer_feats, dim=0)  # [12, 768]
            all_feats.append(layer_feats)

        # [B, 12, 768]
        all_feats = torch.stack(all_feats, dim=0)

        # 加权求和12层
        weights = torch.softmax(self.layer_weights, dim=0)
        feat = (all_feats * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # [B, 768]

        return self.classifier(feat)


# ==========================================
# 训练 & 评估
# ==========================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for face, lhand, rhand, body, labels, lengths in loader:
        face = face.to(device)
        lhand = lhand.to(device)
        rhand = rhand.to(device)
        body = body.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(face, lhand, rhand, body, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += len(labels)

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, device, topk=(1, 5, 10)):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for face, lhand, rhand, body, labels, lengths in loader:
            face = face.to(device)
            lhand = lhand.to(device)
            rhand = rhand.to(device)
            body = body.to(device)
            lengths = lengths.to(device)

            logits = model(face, lhand, rhand, body, lengths)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    results = {}
    for k in topk:
        topk_preds = all_logits.topk(k, dim=1).indices
        correct = topk_preds.eq(all_labels.unsqueeze(1)).any(dim=1).float().mean().item()
        results[f'Rec@{k}'] = correct

    return results


# ==========================================
# Main
# ==========================================
def main(args):
    wandb.init(
        project="islr_lora",
        name=os.path.basename(args.output_dir),
        config=vars(args)
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    with open(args.meta_path, 'r') as f:
        meta = json.load(f)

    all_glosses = sorted(set(item['gloss'] for item in meta))
    label2idx = {g: i for i, g in enumerate(all_glosses)}
    num_classes = len(all_glosses)
    print(f'Total classes: {num_classes}')

    train_dataset = WLASLDataset(args.features_base, meta, 'train', label2idx)
    val_dataset = WLASLDataset(args.features_base, meta, 'val', label2idx)
    test_dataset = WLASLDataset(args.features_base, meta, 'test', label2idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, collate_fn=collate_fn)

    print('Loading SHuBERT with LoRA...')
    shubert = load_shubert_with_lora(args.checkpoint_path)

    model = ISLRLoRA(shubert, num_classes=num_classes).to(device)

    # LoRA 学习率是分类头的 1/10
    optimizer = torch.optim.Adam([
        {'params': model.shubert.parameters(), 'lr': args.lr * 0.1},
        {'params': model.layer_weights, 'lr': args.lr},
        {'params': model.classifier.parameters(), 'lr': args.lr},
    ], weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    os.makedirs(args.output_dir, exist_ok=True)

    best_val_rec1 = 0
    patience = 20
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_results = eval_epoch(model, val_loader, device)

        print(f'Epoch {epoch+1}/{args.epochs} | '
              f'Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.4f} | '
              f'Val Rec@1: {val_results["Rec@1"]:.4f} | '
              f'Val Rec@5: {val_results["Rec@5"]:.4f}')
        wandb.log({
            'epoch': epoch + 1,
            'train/loss': train_loss,
            'train/acc': train_acc,
            'val/Rec@1': val_results['Rec@1'],
            'val/Rec@5': val_results['Rec@5'],
            'val/Rec@10': val_results['Rec@10'],
        })
        if val_results['Rec@1'] > best_val_rec1:
            best_val_rec1 = val_results['Rec@1']
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, 'best_model.pt'))
            print(f'  → Best model saved! Val Rec@1: {best_val_rec1:.4f}')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    print('\nLoading best model for test evaluation...')
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    test_results = eval_epoch(model, test_loader, device)

    print('='*50)
    print('Test Results:')
    for k, v in test_results.items():
        print(f'  {k}: {v:.4f}')
    print('='*50)

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
    parser.add_argument('--checkpoint_path', type=str,
                        default='/scratch/project_465002480/YouTube_ASL/training_logs/run_tensorboard/ckpt/checkpoint_last.pt')
    parser.add_argument('--output_dir', type=str,
                        default='/scratch/project_465002480/YouTube_ASL/wlasl/islr_lora_results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=125)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()
    main(args)
