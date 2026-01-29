"""
Evaluation script for SudokuURM_AR (Autoregressive URM with KV Cache)
"""

import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import numpy as np

from models.urm_ar import SudokuURM_AR, URMConfig
from models.transformer_v3 import VOCAB_SIZE, SOS_TOKEN, action_to_token, is_valid_sudoku 
from dataset.ar_dataset_v3 import SudokuARDatasetV3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(checkpoint_path: str, data_dir: str, split: str = 'test', 
             max_samples: int = None, temperature: float = 0.0):
    
    print(f"Loading URM model from {checkpoint_path}...")
    
    # Load state dict to infer config if possible, or use default URMConfig
    # We assume the standard config used in train_urm.py
    config = URMConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=384,
        num_layers=4,
        n_recurrence=8,
        num_heads=6,
        expansion=4.0,
        max_seq_len=82
    )
    
    model = SudokuURM_AR(config)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    except RuntimeError as e:
        print(f"Error loading checkpoint: {e}")
        print("Attempting to ignore size mismatch (e.g. rope extras)...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=False)

    model.to(DEVICE)
    model.eval()

    print(f"Loading {split} dataset from {data_dir}...")
    dataset = SudokuARDatasetV3(data_dir, split=split, max_samples=max_samples)

    total_puzzles = 0
    solved_puzzles = 0
    total_actions = 0
    correct_actions = 0

    print(f"\nEvaluating on {len(dataset)} samples...")
    print(f"Temperature: {temperature} ({'greedy' if temperature == 0 else 'sampling'})")
    print("-" * 60)

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        item = dataset.data[idx]
        initial_board = item['initial_board']
        oracle_steps = item['steps']  # Oracle order

        # Solution board ground truth
        sol_board = [int(c) for c in initial_board]
        for cell_id, val in oracle_steps:
            sol_board[cell_id] = val

        # Generate (Stateful/Fast)
        # generate_fast is aliased to generate in models/urm_ar.py
        with torch.no_grad():
            pred_actions, pred_board = model.generate(
                initial_board,
                max_actions=len(oracle_steps),
                temperature=temperature,
                device=DEVICE
            )

        # Action Accuracy
        for pred_act, oracle_act in zip(pred_actions, oracle_steps):
            total_actions += 1
            if pred_act == tuple(oracle_act):
                correct_actions += 1
        
        total_actions += max(0, len(oracle_steps) - len(pred_actions))

        # Solve Rate
        total_puzzles += 1
        if pred_board == sol_board and is_valid_sudoku(pred_board):
            solved_puzzles += 1

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (URM - Fast Inference)")
    print("=" * 60)

    solve_rate = solved_puzzles / total_puzzles * 100
    action_acc = correct_actions / total_actions * 100 if total_actions > 0 else 0

    print(f"\n[Primary Metric]")
    print(f"  Solve Rate: {solved_puzzles}/{total_puzzles} = {solve_rate:.2f}%")
    print(f"\n[Secondary Metrics]")
    print(f"  Action Accuracy: {correct_actions}/{total_actions} = {action_acc:.2f}%")
    print("=" * 60)

    return {'solve_rate': solve_rate, 'action_accuracy': action_acc}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/sudoku-trajectory")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    
    evaluate(args.checkpoint, args.data_dir, args.split, args.max_samples, args.temperature)
