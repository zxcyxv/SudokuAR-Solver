"""
Evaluation script for SudokuTransformerV3 (Unified Action Tokens)
"""

import torch
import argparse
from tqdm import tqdm
import numpy as np

from models.transformer_v3 import (
    SudokuTransformerV3, SOS_TOKEN, VOCAB_SIZE,
    action_to_token, token_to_action, is_action_token
)
from dataset.ar_dataset_v3 import SudokuARDatasetV3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def is_valid_sudoku(board: list) -> bool:
    """Check if board is a valid complete sudoku solution."""
    if len(board) != 81 or 0 in board:
        return False

    for r in range(9):
        row = board[r*9:(r+1)*9]
        if len(set(row)) != 9:
            return False

    for c in range(9):
        col = [board[r*9 + c] for r in range(9)]
        if len(set(col)) != 9:
            return False

    for box_r in range(3):
        for box_c in range(3):
            box = []
            for r in range(3):
                for c in range(3):
                    idx = (box_r*3 + r) * 9 + (box_c*3 + c)
                    box.append(board[idx])
            if len(set(box)) != 9:
                return False

    return True


def evaluate(checkpoint_path: str, data_dir: str, split: str = 'test',
             max_samples: int = None, temperature: float = 0.0):
    """
    Evaluate V3 model on Solve Rate and Action Accuracy.
    """
    print(f"Loading model from {checkpoint_path}...")
    model = SudokuTransformerV3(num_layers=4, hidden_dim=384, num_heads=6)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print(f"Loading {split} dataset from {data_dir}...")
    dataset = SudokuARDatasetV3(data_dir, split=split, max_samples=max_samples)

    # Metrics
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
        oracle_steps = item['steps']  # Oracle order: [(cell_id, value), ...]

        # Build solution board
        sol_board = [int(c) for c in initial_board]
        for cell_id, val in oracle_steps:
            sol_board[cell_id] = val

        # Generate with model
        with torch.no_grad():
            pred_actions, pred_board = model.generate(
                initial_board,
                max_actions=len(oracle_steps),
                temperature=temperature,
                device=DEVICE
            )

        # Compare actions (order matters for accuracy)
        for i, (pred_act, oracle_act) in enumerate(zip(pred_actions, oracle_steps)):
            total_actions += 1
            if pred_act == tuple(oracle_act):
                correct_actions += 1

        # Count remaining oracle actions not predicted
        total_actions += max(0, len(oracle_steps) - len(pred_actions))

        # Check solve rate
        total_puzzles += 1
        if pred_board == sol_board and is_valid_sudoku(pred_board):
            solved_puzzles += 1

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (V3 - Unified Action Tokens)")
    print("=" * 60)

    solve_rate = solved_puzzles / total_puzzles * 100
    action_acc = correct_actions / total_actions * 100 if total_actions > 0 else 0

    print(f"\n[Primary Metric]")
    print(f"  Solve Rate: {solved_puzzles}/{total_puzzles} = {solve_rate:.2f}%")

    print(f"\n[Secondary Metrics]")
    print(f"  Action Accuracy (order-sensitive): {correct_actions}/{total_actions} = {action_acc:.2f}%")

    print("=" * 60)

    return {
        'solve_rate': solve_rate,
        'action_accuracy': action_acc,
        'solved': solved_puzzles,
        'total': total_puzzles
    }


def evaluate_cell_accuracy(checkpoint_path: str, data_dir: str, split: str = 'test',
                           max_samples: int = None, temperature: float = 0.0):
    """
    Alternative evaluation: Cell accuracy (order-insensitive).
    """
    print(f"Loading model from {checkpoint_path}...")
    model = SudokuTransformerV3(num_layers=4, hidden_dim=384, num_heads=6)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print(f"Loading {split} dataset from {data_dir}...")
    dataset = SudokuARDatasetV3(data_dir, split=split, max_samples=max_samples)

    total_puzzles = 0
    solved_puzzles = 0
    total_cells = 0
    correct_cells = 0

    print(f"\nEvaluating cell accuracy on {len(dataset)} samples...")

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        item = dataset.data[idx]
        initial_board = item['initial_board']
        oracle_steps = item['steps']

        # Build solution board
        sol_board = [int(c) for c in initial_board]
        for cell_id, val in oracle_steps:
            sol_board[cell_id] = val

        # Generate
        with torch.no_grad():
            _, pred_board = model.generate(
                initial_board,
                max_actions=len(oracle_steps),
                temperature=temperature,
                device=DEVICE
            )

        # Cell accuracy (only empty cells)
        for i in range(81):
            if initial_board[i] == '0':
                total_cells += 1
                if pred_board[i] == sol_board[i]:
                    correct_cells += 1

        # Solve rate
        total_puzzles += 1
        if pred_board == sol_board and is_valid_sudoku(pred_board):
            solved_puzzles += 1

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (V3 - Cell Accuracy)")
    print("=" * 60)

    solve_rate = solved_puzzles / total_puzzles * 100
    cell_acc = correct_cells / total_cells * 100 if total_cells > 0 else 0

    print(f"\n[Primary Metric]")
    print(f"  Solve Rate: {solved_puzzles}/{total_puzzles} = {solve_rate:.2f}%")

    print(f"\n[Secondary Metrics]")
    print(f"  Cell Accuracy: {correct_cells}/{total_cells} = {cell_acc:.2f}%")

    print("=" * 60)

    return {
        'solve_rate': solve_rate,
        'cell_accuracy': cell_acc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Sudoku AR Model V3")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/sudoku-trajectory")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cell_acc", action="store_true", help="Use cell accuracy instead of action accuracy")

    args = parser.parse_args()

    if args.cell_acc:
        evaluate_cell_accuracy(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            split=args.split,
            max_samples=args.max_samples,
            temperature=args.temperature
        )
    else:
        evaluate(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            split=args.split,
            max_samples=args.max_samples,
            temperature=args.temperature
        )
