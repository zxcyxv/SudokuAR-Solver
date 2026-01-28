"""
Dataset V2: Fixed Position Order + Value-Only Prediction

Sequence Format:
----------------
- value_ids: [SOS, V_given1, V_given2, ..., V_empty1, V_empty2, ...]
- board_pos: [0,   pos_given1, pos_given2, ..., pos_empty1, pos_empty2, ...]

Where:
- SOS token at position 0 (arbitrary, could be any position)
- Givens come first (raster scan order)
- Empty cells filled in raster scan order

The model predicts VALUE only. Position is provided as input.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os

from models.transformer_v2 import SOS_TOKEN, val_to_token


class SudokuARDatasetV2(Dataset):
    """
    Dataset for Value-Only Prediction with Fixed Position Order.
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

        print(f"Loaded {len(self.data)} samples from {split} set (V2 format).")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        initial_board = item['initial_board']  # str '003...'
        steps = item['steps']  # list of [pos, val]

        # 1. Parse board
        givens = []  # (pos, val) for given cells
        empty_cells = []  # pos for empty cells (raster order)

        for i, char in enumerate(initial_board):
            if char != '0':
                givens.append((i, int(char)))
            else:
                empty_cells.append(i)

        # 2. Build solution mapping: pos -> val
        solution = {}
        for pos, val in steps:
            solution[pos] = val

        # 3. Build sequences
        # Format: [SOS, given1, given2, ..., empty1, empty2, ...]
        # Position 0 for SOS (arbitrary)

        value_ids = [SOS_TOKEN]
        board_pos = [0]  # SOS position (arbitrary, model will learn to ignore)

        # Add givens (raster order)
        for pos, val in givens:
            value_ids.append(val_to_token(val))
            board_pos.append(pos)

        # Add empty cells in raster order with their solution values
        for pos in empty_cells:
            val = solution.get(pos, 1)  # Default 1 if missing (shouldn't happen)
            value_ids.append(val_to_token(val))
            board_pos.append(pos)

        # 4. Convert to tensors
        value_ids = torch.tensor(value_ids, dtype=torch.long)
        board_pos = torch.tensor(board_pos, dtype=torch.long)

        # 5. Create input/target pairs (shifted by 1)
        # Input:  [SOS, V1, V2, ..., Vn-1]
        # Target: [V1, V2, ..., Vn]
        # BoardPos for input: [pos0, pos1, ..., posn-1]

        value_in = value_ids[:-1]
        value_tgt = value_ids[1:]
        pos_in = board_pos[:-1]

        # Mark where targets are for empty cells (not givens)
        # Givens: positions 1 to len(givens)
        # Empty: positions len(givens)+1 onwards
        n_givens = len(givens)

        return {
            "value_in": value_in,
            "board_pos": pos_in,
            "value_tgt": value_tgt,
            "n_givens": n_givens,  # For masking loss on givens if needed
        }


def collate_fn_v2(batch):
    """
    Collate function for V2 dataset.
    Pads sequences to same length.
    """
    value_in = [item["value_in"] for item in batch]
    board_pos = [item["board_pos"] for item in batch]
    value_tgt = [item["value_tgt"] for item in batch]
    n_givens = [item["n_givens"] for item in batch]

    max_len = max(len(v) for v in value_in)

    padded_value_in = []
    padded_board_pos = []
    padded_value_tgt = []
    attention_mask = []

    for v_in, b_pos, v_tgt in zip(value_in, board_pos, value_tgt):
        pad_len = max_len - len(v_in)
        if pad_len > 0:
            padded_value_in.append(torch.cat([v_in, torch.zeros(pad_len, dtype=torch.long)]))
            padded_board_pos.append(torch.cat([b_pos, torch.zeros(pad_len, dtype=torch.long)]))
            padded_value_tgt.append(torch.cat([v_tgt, torch.zeros(pad_len, dtype=torch.long)]))
            attention_mask.append(torch.cat([torch.ones(len(v_in)), torch.zeros(pad_len)]))
        else:
            padded_value_in.append(v_in)
            padded_board_pos.append(b_pos)
            padded_value_tgt.append(v_tgt)
            attention_mask.append(torch.ones(len(v_in)))

    return {
        "value_in": torch.stack(padded_value_in),
        "board_pos": torch.stack(padded_board_pos),
        "value_tgt": torch.stack(padded_value_tgt),
        "attention_mask": torch.stack(attention_mask),
        "n_givens": torch.tensor(n_givens, dtype=torch.long),
    }


# For quick testing
if __name__ == "__main__":
    ds = SudokuARDatasetV2("data/sudoku-trajectory", split="test", max_samples=5)

    for i in range(min(3, len(ds))):
        sample = ds[i]
        print(f"\nSample {i}:")
        print(f"  value_in shape: {sample['value_in'].shape}")
        print(f"  board_pos shape: {sample['board_pos'].shape}")
        print(f"  value_tgt shape: {sample['value_tgt'].shape}")
        print(f"  n_givens: {sample['n_givens']}")
        print(f"  value_in[:10]: {sample['value_in'][:10].tolist()}")
        print(f"  board_pos[:10]: {sample['board_pos'][:10].tolist()}")
        print(f"  value_tgt[:10]: {sample['value_tgt'][:10].tolist()}")
