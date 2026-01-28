import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import math

from models.transformer import SudokuTransformer
from dataset.ar_dataset import SudokuARDataset

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
    # Correct path: data/sudoku-trajectory
    train_ds = SudokuARDataset("data/sudoku-trajectory", split="train", max_samples=None)
    
    # num_workers=4 for training
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    
    print(f"Dataset Size: {len(train_ds)}")
    print(f"DataLoader Length: {len(train_dl)}")
    
    # 2. Model
    model = SudokuTransformer().to(DEVICE)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 3. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    
    # 4. Scheduler (Cosine)
    total_steps = len(train_dl) * EPOCHS
    if total_steps <= 0:
        total_steps = EPOCHS # Fallback
        
    print(f"Total Steps: {total_steps}")
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps, pct_start=0.1)
    
    # 5. Loop
    criterion_pos = nn.CrossEntropyLoss()
    criterion_val = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
        avg_loss = 0
        
        for batch in pbar:
            pos_in = batch['pos_in'].to(DEVICE)
            val_in = batch['val_in'].to(DEVICE)
            pos_tgt = batch['pos_tgt'].to(DEVICE)
            val_tgt = batch['val_tgt'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            logits_pos, logits_val = model(pos_in, val_in)
            
            # Loss: Flatten for CE
            # logits: [b, s, classes] -> [b*s, classes]
            # targets: [b, s] -> [b*s]
            loss_pos = criterion_pos(logits_pos.view(-1, 81), pos_tgt.view(-1))
            loss_val = criterion_val(logits_val.view(-1, 10), val_tgt.view(-1))
            
            loss = loss_pos + loss_val
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            
            avg_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "pos": f"{loss_pos.item():.2f}", "val": f"{loss_val.item():.2f}"})
            
        print(f"Epoch {epoch+1} Average Loss: {avg_loss/len(train_dl):.4f}")
        
        # Save
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), f"checkpoints/ar_trm_ep{epoch+1}.pth")

if __name__ == "__main__":
    train()
