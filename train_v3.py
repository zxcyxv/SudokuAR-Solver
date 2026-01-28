"""
Training script for SudokuTransformerV3 (Unified Action Tokens + Oracle Order)

Key features:
- Single token = Position + Value combined
- Oracle order (easiest cells first)
- Standard next-token prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models.transformer_v3 import SudokuTransformerV3, VOCAB_SIZE, SOS_TOKEN
from dataset.ar_dataset_v3 import SudokuARDatasetV3, collate_fn_v3

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
    train_ds = SudokuARDatasetV3("data/sudoku-trajectory", split="train", max_samples=None)

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn_v3
    )

    print(f"Dataset Size: {len(train_ds)}")
    print(f"DataLoader Length: {len(train_dl)}")

    # 2. Model
    model = SudokuTransformerV3(
        num_layers=4,
        hidden_dim=384,
        num_heads=6,
    ).to(DEVICE)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Vocab Size: {VOCAB_SIZE} (729 actions + 1 SOS)")

    # 3. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

    # 4. Scheduler
    total_steps = len(train_dl) * EPOCHS
    print(f"Total Steps: {total_steps}")

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps, pct_start=0.1
    )

    # 5. Loss function (ignore SOS_TOKEN used for padding)
    criterion = nn.CrossEntropyLoss(ignore_index=SOS_TOKEN)

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for batch in pbar:
            token_in = batch['token_in'].to(DEVICE)
            token_tgt = batch['token_tgt'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            optimizer.zero_grad()

            # Forward
            logits = model(token_in)  # [b, seq_len, vocab_size]

            # Compute loss
            loss = criterion(
                logits.view(-1, VOCAB_SIZE),
                token_tgt.view(-1)
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            # Compute accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = attention_mask.bool()
                correct = ((preds == token_tgt) & mask).sum().item()
                total = mask.sum().item()

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
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")

        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoints/ar_trm_v3_ep{epoch+1}.pth")


if __name__ == "__main__":
    train()
