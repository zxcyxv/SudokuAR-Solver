"""
Dataset V3: Unified Action Tokens with Oracle Order

Token Design:
- Token = Cell_ID * 9 + (Value - 1)
- Range: 0-728 (729 action tokens)
- SOS: 729

Sequence Format:
[SOS, Given1, Given2, ..., GivenN, Step1, Step2, ..., StepM]

Where:
- Givens are in RASTER order (0->80)
- Steps are in ORACLE order (easiest cells first)

The model learns to predict Steps given Givens as context.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os

from models.transformer_v3 import SOS_TOKEN, action_to_token, VOCAB_SIZE


class SudokuARDatasetV3(Dataset):
    """
    Dataset with Unified Action Tokens.
    Includes Givens (raster order) + Steps (Oracle order).
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

        print(f"Loaded {len(self.data)} samples from {split} set (V3 format).")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        initial_board = item['initial_board']
        steps = item['steps']  # Oracle order: [(cell_id, value), ...]

        # 1. Build Givens tokens (raster order: 0->80)
        given_tokens = []
        for cell_id, char in enumerate(initial_board):
            if char != '0':
                value = int(char)
                token = action_to_token(cell_id, value)
                given_tokens.append(token)

        # 2. Build Steps tokens (Oracle order)
        step_tokens = []
        for cell_id, value in steps:
            token = action_to_token(cell_id, value)
            step_tokens.append(token)

        # 3. Full sequence: [SOS, Givens..., Steps...]
        tokens = [SOS_TOKEN] + given_tokens + step_tokens
        tokens = torch.tensor(tokens, dtype=torch.long)

        # 4. Input/Target pairs (shifted by 1)
        token_in = tokens[:-1]
        token_tgt = tokens[1:]

        # 5. Mark where steps start (for loss masking if needed)
        n_givens = len(given_tokens)

        return {
            "token_in": token_in,
            "token_tgt": token_tgt,
            "n_givens": n_givens,
        }


def collate_fn_v3(batch):
    """
    Collate function for V3 dataset.
    Pads sequences to same length.
    """
    token_in = [item["token_in"] for item in batch]
    token_tgt = [item["token_tgt"] for item in batch]
    n_givens = [item["n_givens"] for item in batch]

    max_len = max(len(t) for t in token_in)

    padded_token_in = []
    padded_token_tgt = []
    attention_mask = []

    for t_in, t_tgt in zip(token_in, token_tgt):
        pad_len = max_len - len(t_in)
        if pad_len > 0:
            padded_token_in.append(torch.cat([t_in, torch.full((pad_len,), SOS_TOKEN, dtype=torch.long)]))
            padded_token_tgt.append(torch.cat([t_tgt, torch.full((pad_len,), SOS_TOKEN, dtype=torch.long)]))
            attention_mask.append(torch.cat([torch.ones(len(t_in)), torch.zeros(pad_len)]))
        else:
            padded_token_in.append(t_in)
            padded_token_tgt.append(t_tgt)
            attention_mask.append(torch.ones(len(t_in)))

    return {
        "token_in": torch.stack(padded_token_in),
        "token_tgt": torch.stack(padded_token_tgt),
        "attention_mask": torch.stack(attention_mask),
        "n_givens": torch.tensor(n_givens, dtype=torch.long),
    }


# For quick testing
if __name__ == "__main__":
    from models.transformer_v3 import token_to_action

    ds = SudokuARDatasetV3("data/sudoku-trajectory", split="test", max_samples=1)
    sample = ds[0]
    item = ds.data[0]

    print(f"=== V3 Dataset Verification ===")
    print(f"Puzzle: {item['initial_board'][:20]}...")

    givens_count = sum(1 for c in item['initial_board'] if c != '0')
    steps_count = len(item['steps'])

    print(f"\nGivens: {givens_count}")
    print(f"Steps (Oracle): {steps_count}")
    print(f"n_givens in sample: {sample['n_givens']}")

    print(f"\nSequence length: {len(sample['token_in'])} + 1 = {len(sample['token_in']) + 1}")
    print(f"Expected: SOS(1) + Givens({givens_count}) + Steps({steps_count}) = {1 + givens_count + steps_count}")

    # Verify first few tokens are Givens
    print(f"\nFirst 5 tokens (should be Givens):")
    for i, tok in enumerate(sample['token_tgt'][:5].tolist()):
        cell_id, value = token_to_action(tok)
        print(f"  Token {tok} -> Cell {cell_id}, Value {value}")
