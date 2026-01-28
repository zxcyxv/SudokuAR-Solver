import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os

class SudokuARDataset(Dataset):
    def __init__(self, data_dir, split='train', max_samples=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, f"{split}_trajectories_part_*.npy")))
        if not self.files:
            # Fallback to single file if chunking wasn't used/completed fully in a way that matches glob
            single = os.path.join(data_dir, f"{split}_trajectories.npy")
            if os.path.exists(single):
                self.files = [single]
        
        # Load all data into memory for 'Tiny' complexity
        # For larger datasets, one would index differently.
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
        initial_board = item['initial_board'] # str '003...'
        steps = item['steps'] # list of [pos, val]
        
        # 1. Parse Givens
        # Givens are fixed, but we feed them as the start of the sequence.
        # Order: Raster scan (0..80) or whatever. 
        # Standard: Raster scan ensuring deterministic prefix.
        givens = []
        for i, char in enumerate(initial_board):
            if char != '0':
                givens.append([i, int(char)])
        
        # 2. Combine with Steps
        # Sequence: [Start Token?] -> Givens -> Steps
        # User spec: "Input: (Position, Value) ... (Index, Value)"
        # Does not explicitly ask for SOS token.
        # But 'Value Index: 0 (Start)' is mentioned.
        # Let's add a SOS token: Pos=0(dummy), Val=0(SOS).
        # Actually, Pos=81 (unknown) might be better? Or just use Val=0 as distinct.
        # Let's try without SOS first, just Givens + Steps.
        # Input: [Givens..., Steps...]
        # But wait, AR model needs to predict the *next*.
        # For the FIRST given, what is the context? Empty?
        # So we almost certainly need a specific Start Token.
        # Spec says: "Value_Index: 1~9 (number), 0 (Start token etc)".
        # Let's prepend (0, 0) as SOS. (Pos=0 might imply index 0, so maybe use a dummy pos?)
        # Since Pos is 0-80, using 0 is ambiguous.
        # But the Spec says: "Position_Index: 0~80". No special token for Pos mentioned.
        # Unless we use a special value for 'Start'. 
        
        # Let's use:
        # SOS Token: Pos=0, Val=0. (Overloading Pos 0 is fine if Val 0 distinguishes it).
        full_seq = [[0, 0]] + givens + steps
        
        # Truncate to max len? 81 cells max + 1 SOS = 82?
        # A full board has 81 cells. Sequence length is always 82?
        # Ideally yes.
        
        # Convert to tensor
        full_seq = torch.tensor(full_seq, dtype=torch.long)
        
        # Split inputs and targets
        # Inputs: SOS -> Last Step
        # Targets: First Step -> End
        
        inputs = full_seq[:-1]
        targets = full_seq[1:]
        
        # targets contain (pos, val).
        
        # Pad if necessary? (Unlikely if solved fully, always 81 steps + SOS)
        # But wait, steps might be partial if dataset had errors?
        # Assuming full trajectories.
        if len(inputs) < 82:
             # Pad? Or just return variable length? Batching requires same length.
             # Sudoku is fixed size 81.
             pass

        return {
            "pos_in": inputs[:, 0],
            "val_in": inputs[:, 1],
            "pos_tgt": targets[:, 0],
            "val_tgt": targets[:, 1]
        }
