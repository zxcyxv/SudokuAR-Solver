
"""
Training script for SudokuURM_AR (Autoregressive URM)

Configuration:
1. Base: Stable Code (Batch 256, OneCycleLR, Last Step Only)
2. [NEW] Fact-Aware Loss: Checks against the ground truth board.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse

# Import verification utils
from models.transformer_v3 import VOCAB_SIZE, SOS_TOKEN, is_valid_sudoku, action_to_token
from models.urm_ar import SudokuURM_AR, URMConfig
from models.muon import Muon
from dataset.ar_dataset_v3 import SudokuARDatasetV3, collate_fn_v3

# [OPTIMIZATION] Enable TF32 for faster matmul on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

# Hyperparams (Preserved)
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3            # This acts as the Global Max LR for OneCycle
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_SAMPLES = 100

# --- [NEW] Fact-Aware Loss Function ---
def get_fact_aware_loss(logits, targets, solutions, criterion):
    """
    logits: [B, Seq, Vocab]
    targets: [B, Seq] (Dataset's strict order)
    solutions: [B, 81] (Full Ground Truth Board)
    """
    # 1. Standard Cross Entropy (Teacher Forcing Order)
    ce_loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1), reduction='none')
    ce_loss = ce_loss.view(targets.shape) # [B, Seq]

    # 2. Fact Check
    with torch.no_grad():
        preds = logits.argmax(dim=-1) # [B, Seq]
        
        # Parse Tokens
        is_action = (preds < 729) # Ignore SOS(729)
        pred_cells = preds // 9   # 0~80
        pred_vals = (preds % 9) + 1 # 1~9
        
        # Retrieve Truth
        safe_cells = pred_cells.clamp(0, 80) # Safety for gather
        true_vals_at_pred_cell = torch.gather(solutions, 1, safe_cells)
        
        # Check: (Model Value == Truth Value) AND (Is Action Token)
        is_factually_correct = (pred_vals == true_vals_at_pred_cell) & is_action
        
    # 3. Masking
    # If factually correct, mask loss to 0. Otherwise keep CE loss.
    mask = (~is_factually_correct).float()
    
    masked_loss = ce_loss * mask
    
    # Average over valid tokens (add epsilon to avoid div by zero)
    return masked_loss.sum() / (mask.sum() + 1e-6)

# --- [FIXED] Dataset Patching for Solutions ---
original_getitem = SudokuARDatasetV3.__getitem__

def patched_getitem(self, idx):
    # 1. Get the tensors (token_in, token_tgt, etc.)
    item = original_getitem(self, idx)
    
    # 2. [FIX] Access raw data from self.data instead of item
    # 'item' only contains tensors processed by getitem, not the raw keys.
    raw_data = self.data[idx]
    
    # 3. Generate Solution Tensor from raw data
    initial = [int(c) for c in raw_data['initial_board']]
    sol = list(initial)
    for c, v in raw_data['steps']:
        sol[c] = v
        
    item['solution'] = torch.tensor(sol, dtype=torch.long)
    return item

SudokuARDatasetV3.__getitem__ = patched_getitem

def patched_collate_fn(batch):
    out = collate_fn_v3(batch)
    out['solution'] = torch.stack([b['solution'] for b in batch])
    return out


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
        collate_fn=patched_collate_fn # Use patched collate
    )

    print(f"Loading {EVAL_SAMPLES} test samples for evaluation...")
    val_ds = SudokuARDatasetV3("data/sudoku-trajectory", split="test", max_samples=EVAL_SAMPLES)

    print(f"Train Dataset Size: {len(train_ds)}")
    print(f"Val Dataset Size: {len(val_ds)}")

    # 2. Model
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
    
    # [OPTIMIZATION] Compile model
    print("Compiling model with torch.compile...")
    model = torch.compile(model, mode="reduce-overhead")
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 3. Optimizer (Muon + AdamW)
    muon_params = []
    adam_params = []
    
    # Exclude embeddings from Muon
    embed_param_ids = {id(p) for m in [model.token_emb, model.loop_emb] for p in m.parameters()}
    
    for p in model.parameters():
        if p.requires_grad:
            if p.ndim == 2 and id(p) not in embed_param_ids:
                muon_params.append(p)
            else:
                adam_params.append(p)
    
    # Note: OneCycleLR will override these 'lr' values with its own schedule based on max_lr=LR(1e-3)
    optimizer = Muon([
        {"params": muon_params, "use_muon": True, "lr": 0.02, "momentum": 0.95, "adamw_betas": (0.9, 0.95)},
        {"params": adam_params, "use_muon": False, "lr": 1e-3, "weight_decay": WEIGHT_DECAY, "adamw_betas": (0.9, 0.95)}
    ])

    # 4. Scheduler (Restored OneCycleLR)
    total_steps = len(train_dl) * EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LR, 
        total_steps=total_steps, 
        pct_start=0.1
    )

    for epoch in range(EPOCHS):
        # --- Training Loop ---
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        epoch_loss = 0
        epoch_correct = 0
        epoch_tokens = 0

        for batch in pbar:
            token_in = batch['token_in'].to(DEVICE)
            token_tgt = batch['token_tgt'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            solutions = batch['solution'].to(DEVICE) # [NEW] Ground Truth Board

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE=="cuda")):
                outputs = model(token_in) # List[Tensor]
                
                # Last Step Only Logic
                final_logits = outputs[-1]
                
                # [NEW] Use Fact-Aware Loss
                loss = get_fact_aware_loss(final_logits, token_tgt, solutions, None)

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Metrics
            with torch.no_grad():
                preds = final_logits.argmax(dim=-1)
                mask = attention_mask.bool()
                # 'correct' here means matching the strict dataset order
                correct = ((preds == token_tgt) & mask).sum().item()
                total = mask.sum().item()

            epoch_loss += loss.item() * total
            epoch_correct += correct
            epoch_tokens += total

            curr_acc = correct / total * 100 if total > 0 else 0
            
            # Monitoring LR
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "strict_acc": f"{curr_acc:.1f}%", # Label changed to strict_acc
                "lr": f"{current_lr:.5f}"
            })

        avg_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
        avg_acc = epoch_correct / epoch_tokens * 100 if epoch_tokens > 0 else 0
        print(f"Epoch {epoch+1} Summary - Loss: {avg_loss:.4f}, Strict Train Acc: {avg_acc:.2f}%")

        # --- Evaluation Loop ---
        print(f"Running Evaluation on {len(val_ds)} samples...")
        model.eval()
        solved_count = 0
        
        with torch.no_grad():
            for i in tqdm(range(len(val_ds)), desc=f"Epoch {epoch+1} [Eval]"):
                item = val_ds.data[i]
                initial_board = item['initial_board']
                oracle_steps = item['steps']
                
                sol_board = [int(c) for c in initial_board]
                for cell_id, val in oracle_steps:
                    sol_board[cell_id] = val
                
                _, pred_board = model.generate(
                    initial_board, 
                    max_actions=len(oracle_steps), 
                    temperature=0.0, # Greedy
                    device=DEVICE
                )
                
                if pred_board == sol_board and is_valid_sudoku(pred_board):
                    solved_count += 1
        
        solve_rate = solved_count / len(val_ds) * 100
        print(f"Epoch {epoch+1} EVAL RESULT -> Solve Rate: {solve_rate:.2f}% ({solved_count}/{len(val_ds)})")
        print("-" * 60)

        torch.save(model.state_dict(), f"checkpoints/ar_urm_v3_ep{epoch+1}.pth")


if __name__ == "__main__":
    train()
