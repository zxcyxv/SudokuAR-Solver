import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os

# Import token utilities from model
from models.transformer import (
    SOS_TOKEN, pos_to_token, val_to_token,
    POS_TOKEN_OFFSET, VAL_TOKEN_OFFSET
)


class SudokuARDataset(Dataset):
    """
    Dataset for Serialized Autoregressive Sudoku Solver.

    Sequence Format (Flattened):
    ----------------------------
    [SOS, P_0, V_0, P_1, V_1, ..., P_80, V_80]

    Where:
        - SOS: Start token (token 0)
        - P_i: Position token (tokens 1-81, representing board index 0-80)
        - V_i: Value token (tokens 82-90, representing digits 1-9)

    This design satisfies the Chain Rule of Probability:
        P(C, V | S) = P(C | S) × P(V | C, S)

    When predicting V_i, the model has already seen P_i in its context.
    """

    def __init__(self, data_dir, split='train', max_samples=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, f"{split}_trajectories_part_*.npy")))
        if not self.files:
            # Fallback to single file if chunking wasn't used/completed fully
            single = os.path.join(data_dir, f"{split}_trajectories.npy")
            if os.path.exists(single):
                self.files = [single]

        # Load all data into memory
        self.data = []
        for f in self.files:
            try:
                chunk = np.load(f, allow_pickle=True)
                self.data.extend(chunk)
                if max_samples and len(self.data) >= max_samples:
                    self.data = self.data[:max_samples]
                    break
            except Exception as e:
                print(f"Error loading {f}: {e}")

        print(f"Loaded {len(self.data)} samples from {split} set.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        initial_board = item['initial_board']  # str '003...'
        steps = item['steps']  # list of [pos, val]

        # 1. Parse Givens (in raster scan order for deterministic prefix)
        givens = []
        for i, char in enumerate(initial_board):
            if char != '0':
                givens.append((i, int(char)))

        # 2. Build flattened token sequence
        # Format: [SOS, P_0, V_0, P_1, V_1, ...]
        tokens = [SOS_TOKEN]

        # Add givens (pos, val pairs)
        for pos, val in givens:
            tokens.append(pos_to_token(pos))  # Position token (1-81)
            tokens.append(val_to_token(val))  # Value token (82-90)

        # Add solver steps (pos, val pairs)
        for pos, val in steps:
            tokens.append(pos_to_token(pos))
            tokens.append(val_to_token(val))

        # 3. Convert to tensor
        tokens = torch.tensor(tokens, dtype=torch.long)

        # 4. Create input/target pairs (shifted by 1)
        # Input:  [SOS, P_0, V_0, P_1, V_1, ..., P_n-1, V_n-1]
        # Target: [P_0, V_0, P_1, V_1, ..., P_n, V_n]
        inputs = tokens[:-1]
        targets = tokens[1:]

        # Expected sequence length: 1 (SOS) + 81*2 (pos-val pairs) = 163
        # After shift: inputs and targets each have 162 tokens

        return {
            "input_ids": inputs,
            "target_ids": targets
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.

    For fully solved Sudoku puzzles, all sequences should have the same length,
    but this handles edge cases gracefully.
    """
    input_ids = [item["input_ids"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]

    # Find max length
    max_len = max(len(ids) for ids in input_ids)

    # Pad sequences (using SOS_TOKEN=0 as padding, which is fine since
    # we use a causal mask and don't compute loss on padded positions)
    padded_inputs = []
    padded_targets = []
    attention_mask = []

    for inp, tgt in zip(input_ids, target_ids):
        pad_len = max_len - len(inp)
        if pad_len > 0:
            padded_inputs.append(torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)]))
            padded_targets.append(torch.cat([tgt, torch.zeros(pad_len, dtype=torch.long)]))
            attention_mask.append(torch.cat([torch.ones(len(inp)), torch.zeros(pad_len)]))
        else:
            padded_inputs.append(inp)
            padded_targets.append(tgt)
            attention_mask.append(torch.ones(len(inp)))

    return {
        "input_ids": torch.stack(padded_inputs),
        "target_ids": torch.stack(padded_targets),
        "attention_mask": torch.stack(attention_mask)
    }


# Legacy dataset for comparison (keeps old format)
class SudokuARDatasetLegacy(Dataset):
    """
    Legacy dataset format with parallel (pos, val) targets.
    WARNING: This pairs with the legacy model that has the independence assumption error.
    """

    def __init__(self, data_dir, split='train', max_samples=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, f"{split}_trajectories_part_*.npy")))
        if not self.files:
            single = os.path.join(data_dir, f"{split}_trajectories.npy")
            if os.path.exists(single):
                self.files = [single]

        self.data = []
        for f in self.files:
            try:
                chunk = np.load(f, allow_pickle=True)
                self.data.extend(chunk)
                if max_samples and len(self.data) >= max_samples:
                    self.data = self.data[:max_samples]
                    break
            except Exception as e:
                print(f"Error loading {f}: {e}")

        print(f"Loaded {len(self.data)} samples from {split} set (legacy format).")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        initial_board = item['initial_board']
        steps = item['steps']

        givens = []
        for i, char in enumerate(initial_board):
            if char != '0':
                givens.append([i, int(char)])

        full_seq = [[0, 0]] + givens + steps
        full_seq = torch.tensor(full_seq, dtype=torch.long)

        inputs = full_seq[:-1]
        targets = full_seq[1:]

        return {
            "pos_in": inputs[:, 0],
            "val_in": inputs[:, 1],
            "pos_tgt": targets[:, 0],
            "val_tgt": targets[:, 1]
        }
