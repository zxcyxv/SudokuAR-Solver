"""
Evaluation script for SudokuTransformerV2 (Fixed Position + Value-Only)
"""

import torch
import argparse
from tqdm import tqdm
import numpy as np

from models.transformer_v2 import SudokuTransformerV2, SOS_TOKEN, val_to_token, token_to_val
from dataset.ar_dataset_v2 import SudokuARDatasetV2

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
    Evaluate V2 model on Solve Rate and Cell Accuracy.
    """
    print(f"Loading model from {checkpoint_path}...")
    model = SudokuTransformerV2(num_layers=4, hidden_dim=384, num_heads=6)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print(f"Loading {split} dataset from {data_dir}...")
    dataset = SudokuARDatasetV2(data_dir, split=split, max_samples=max_samples)

    # Metrics
    total = 0
    solved = 0
    total_cells = 0
    correct_cells = 0

    print(f"\nEvaluating on {len(dataset)} samples...")
    print(f"Temperature: {temperature} ({'greedy' if temperature == 0 else 'sampling'})")
    print("-" * 60)

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        item = dataset.data[idx]
        initial_board = item['initial_board']
        steps = item['steps']

        # Build solution board
        sol_board = [int(c) for c in initial_board]
        for pos, val in steps:
            sol_board[pos] = val

        # Parse board
        givens = []
        empty_cells = []
        for i, char in enumerate(initial_board):
            if char != '0':
                givens.append((i, int(char)))
            else:
                empty_cells.append(i)

        # Build initial sequence (SOS + givens)
        initial_values = [SOS_TOKEN]
        initial_pos = [0]
        for pos, val in givens:
            initial_values.append(val_to_token(val))
            initial_pos.append(pos)

        initial_values = torch.tensor([initial_values], dtype=torch.long, device=DEVICE)
        initial_pos = torch.tensor([initial_pos], dtype=torch.long, device=DEVICE)
        remaining_pos = torch.tensor([empty_cells], dtype=torch.long, device=DEVICE)

        # Generate
        with torch.no_grad():
            if temperature == 0:
                # Greedy generation
                values = initial_values.clone()
                positions = initial_pos.clone()
                generated = []

                for i in range(len(empty_cells)):
                    next_pos = remaining_pos[:, i:i+1]
                    logits = model(values, positions)
                    next_logits = logits[:, -1, :]
                    next_logits[:, SOS_TOKEN] = float('-inf')
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                    generated.append(next_token)
                    values = torch.cat([values, next_token], dim=1)
                    positions = torch.cat([positions, next_pos], dim=1)

                pred_values = torch.cat(generated, dim=1)[0].cpu().tolist()
            else:
                pred_values = model.generate(
                    initial_values, initial_pos, remaining_pos,
                    temperature=temperature
                )[0].cpu().tolist()

        # Build predicted board
        pred_board = [int(c) for c in initial_board]
        for i, pos in enumerate(empty_cells):
            if i < len(pred_values):
                pred_board[pos] = token_to_val(pred_values[i])

        # Compute metrics
        for pos in empty_cells:
            total_cells += 1
            if pred_board[pos] == sol_board[pos]:
                correct_cells += 1

        total += 1
        if pred_board == sol_board and is_valid_sudoku(pred_board):
            solved += 1

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (V2 - Fixed Position)")
    print("=" * 60)

    solve_rate = solved / total * 100
    cell_acc = correct_cells / total_cells * 100 if total_cells > 0 else 0

    print(f"\n[Primary Metric]")
    print(f"  Solve Rate: {solved}/{total} = {solve_rate:.2f}%")

    print(f"\n[Secondary Metrics]")
    print(f"  Cell Accuracy (empty cells): {correct_cells}/{total_cells} = {cell_acc:.2f}%")

    print(f"\n[Dataset Statistics]")
    print(f"  Avg empty cells: {total_cells/total:.1f}")

    print("=" * 60)

    return {
        'solve_rate': solve_rate,
        'cell_accuracy': cell_acc,
        'solved': solved,
        'total': total
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Sudoku AR Model V2")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/sudoku-trajectory",
                        help="Path to dataset directory")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (train/test)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")

    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        max_samples=args.max_samples,
        temperature=args.temperature
    )
