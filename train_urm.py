"""
Training script for SudokuURM_AR (Autoregressive URM)

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
import argparse # Added argparse

from models.transformer_v3 import VOCAB_SIZE, SOS_TOKEN
from models.urm_ar import SudokuURM_AR, URMConfig
from models.muon import Muon
from dataset.ar_dataset_v3 import SudokuARDatasetV3, collate_fn_v3

# [OPTIMIZATION] Enable TF32 for faster matmul on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

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
    # 2. Model (URM)
    config = URMConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=384,
        num_layers=4,       # Physical layers
        n_recurrence=8,     # Inner Loop count (Recurrence)
        num_heads=6,
        expansion=4.0,
        max_seq_len=82
    )
    model = SudokuURM_AR(config).to(DEVICE)
    
    # [OPTIMIZATION] Compile model to fuse kernels (Gate->Up->Mul->Pad->Conv->Act->Down)
    # mode="reduce-overhead" is best for short sequences (Sudoku: 82)
    print("Compiling model with torch.compile...")
    model = torch.compile(model, mode="reduce-overhead")
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Vocab Size: {VOCAB_SIZE} (729 actions + 1 SOS)")

    # 3. Optimizer (Revised)
    # Filter params: Muon for 2D weights (Linear, Conv kernel if 2D), but EXCLUDE Embeddings
    
    muon_params = []
    adam_params = []
    
    # Identify embedding parameters by ID to exclude them from Muon
    # (Token Embedding + Loop Embedding)
    embed_param_ids = {id(p) for m in [model.token_emb, model.loop_emb] for p in m.parameters()}
    
    for p in model.parameters():
        if p.requires_grad:
            # Use Muon only for 2D weights that are NOT embeddings
            if p.ndim == 2 and id(p) not in embed_param_ids:
                muon_params.append(p)
            else:
                adam_params.append(p)
    
    optimizer = Muon([
        {
            "params": muon_params,
            "use_muon": True,
            "lr": 0.02, # Default Muon LR
            "momentum": 0.95,
            "adamw_betas": (0.9, 0.95),
        },
        {
            "params": adam_params,
            "use_muon": False,
            "lr": 1e-3, # Keep original AdamW LR for embeddings/biases
            "weight_decay": WEIGHT_DECAY,
            "adamw_betas": (0.9, 0.95),
        }
    ])

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
        
        # [Epoch Accumulators]
        epoch_loss = 0    # Variable rename (total_loss -> epoch_loss)
        epoch_correct = 0 # Variable rename (total_correct -> epoch_correct)
        epoch_tokens = 0  # Variable rename (total_tokens -> epoch_tokens)

        for batch in pbar:
            token_in = batch['token_in'].to(DEVICE)
            token_tgt = batch['token_tgt'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            # [OPTIMIZATION] Mixed Precision (BFloat16)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE=="cuda")):
                outputs = model(token_in)  # List[Tensor] (Deep Supervision)
                
                # Deep Supervision (Sum of Losses)
                batch_loss = 0 # [FIX 1] Avoid name collision
                for step_logits in outputs:
                    step_loss = criterion(
                        step_logits.view(-1, VOCAB_SIZE),
                        token_tgt.view(-1)
                    )
                    batch_loss += step_loss
                
                loss = batch_loss 

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Compute accuracy
            with torch.no_grad():
                # [FIX 2] Get last element from outputs list
                final_logits = outputs[-1] 
                
                preds = final_logits.argmax(dim=-1)
                mask = attention_mask.bool()
                correct = ((preds == token_tgt) & mask).sum().item()
                total = mask.sum().item()

            # Epoch Accumulation
            epoch_loss += loss.item() * total 
            epoch_correct += correct
            epoch_tokens += total

            # Logging
            curr_acc = correct / total * 100 if total > 0 else 0
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{curr_acc:.1f}%"
            })

        # Epoch Summary
        avg_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
        avg_acc = epoch_correct / epoch_tokens * 100 if epoch_tokens > 0 else 0
        
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2f}%")

        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoints/ar_urm_v3_ep{epoch+1}.pth")


if __name__ == "__main__":
    train()
