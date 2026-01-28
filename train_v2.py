"""
Training script for SudokuTransformerV2 (Fixed Position + Value-Only)

Key fix: Loss is computed ONLY on empty cells, not on givens.
This prevents gradient dilution from easy given predictions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models.transformer_v2 import SudokuTransformerV2, VOCAB_SIZE
from dataset.ar_dataset_v2 import SudokuARDatasetV2, collate_fn_v2

# Hyperparams
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    os.makedirs("checkpoints", exist_ok=True)

    # 1. Dataset
    train_ds = SudokuARDatasetV2("data/sudoku-trajectory", split="train", max_samples=None)

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn_v2
    )

    print(f"Dataset Size: {len(train_ds)}")
    print(f"DataLoader Length: {len(train_dl)}")

    # 2. Model
    model = SudokuTransformerV2(
        num_layers=4,
        hidden_dim=384,
        num_heads=6,
    ).to(DEVICE)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 3. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

    # 4. Scheduler
    total_steps = len(train_dl) * EPOCHS
    if total_steps <= 0:
        total_steps = EPOCHS

    print(f"Total Steps: {total_steps}")

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps, pct_start=0.1)

    # 5. Loss function (reduction='none' for custom masking)
    criterion = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for batch in pbar:
            value_in = batch['value_in'].to(DEVICE)
            board_pos = batch['board_pos'].to(DEVICE)
            value_tgt = batch['value_tgt'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            n_givens = batch['n_givens'].to(DEVICE)

            batch_size, seq_len = value_in.shape

            # Create empty cell mask (positions after n_givens for each sample)
            empty_mask = torch.zeros(batch_size, seq_len, device=DEVICE)
            for b in range(batch_size):
                ng = n_givens[b].item()
                empty_mask[b, ng:] = attention_mask[b, ng:]

            optimizer.zero_grad()

            # Forward
            logits = model(value_in, board_pos)  # [b, seq_len, vocab_size]

            # Compute loss ONLY on empty cells (not givens)
            raw_loss = criterion(
                logits.view(-1, VOCAB_SIZE),
                value_tgt.view(-1)
            ).view(batch_size, seq_len)

            loss = (raw_loss * empty_mask).sum() / (empty_mask.sum() + 1e-6)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            # Compute accuracy on empty cells
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                empty_mask_bool = empty_mask.bool()
                correct = ((preds == value_tgt) & empty_mask_bool).sum().item()
                total = empty_mask.sum().item()

            total_loss += loss.item() * total
            total_correct += correct
            total_tokens += total

            acc = correct / total * 100 if total > 0 else 0
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.1f}%"
            })

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        avg_acc = total_correct / total_tokens * 100 if total_tokens > 0 else 0
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Empty Cell Acc: {avg_acc:.2f}%")

        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoints/ar_trm_v2_ep{epoch+1}.pth")


if __name__ == "__main__":
    train()
