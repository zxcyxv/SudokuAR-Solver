import torch
import argparse
from tqdm import tqdm
import numpy as np

from models.transformer import (
    SudokuTransformer, SOS_TOKEN, VOCAB_SIZE,
    pos_to_token, val_to_token, token_to_pos, token_to_val,
    is_pos_token, is_val_token
)
from dataset.ar_dataset import SudokuARDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_initial_tokens(initial_board: str) -> torch.Tensor:
    """
    Build initial token sequence from puzzle string.
    Returns: [SOS, P0, V0, P1, V1, ...] for givens only
    """
    tokens = [SOS_TOKEN]
    for i, char in enumerate(initial_board):
        if char != '0':
            tokens.append(pos_to_token(i))
            tokens.append(val_to_token(int(char)))
    return torch.tensor(tokens, dtype=torch.long)


def decode_sequence_to_board(tokens: list, initial_board: str) -> list:
    """
    Decode token sequence to 81-cell board.
    Returns: List of 81 integers (0 for empty, 1-9 for filled)
    """
    # Start with initial board
    board = [int(c) for c in initial_board]

    # Parse generated tokens (skip SOS at index 0)
    i = 1
    while i + 1 < len(tokens):
        pos_tok = tokens[i]
        val_tok = tokens[i + 1]

        if is_pos_token(pos_tok) and is_val_token(val_tok):
            pos = token_to_pos(pos_tok)
            val = token_to_val(val_tok)
            if 0 <= pos < 81 and 1 <= val <= 9:
                board[pos] = val
        i += 2

    return board


def is_valid_sudoku(board: list) -> bool:
    """Check if board is a valid complete sudoku solution."""
    if len(board) != 81 or 0 in board:
        return False

    # Check rows
    for r in range(9):
        row = board[r*9:(r+1)*9]
        if len(set(row)) != 9:
            return False

    # Check columns
    for c in range(9):
        col = [board[r*9 + c] for r in range(9)]
        if len(set(col)) != 9:
            return False

    # Check 3x3 boxes
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
    Evaluate model on Solve Rate and Cell Accuracy.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to dataset directory
        split: 'train' or 'test'
        max_samples: Limit number of samples (None for all)
        temperature: Sampling temperature (0 = greedy)
    """
    print(f"Loading model from {checkpoint_path}...")
    model = SudokuTransformer()
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print(f"Loading {split} dataset from {data_dir}...")
    dataset = SudokuARDataset(data_dir, split=split, max_samples=max_samples)

    # Metrics
    total = 0
    solved = 0
    total_cells = 0
    correct_cells = 0

    # For detailed analysis
    empty_cells_list = []
    correct_per_puzzle = []

    print(f"\nEvaluating on {len(dataset)} samples...")
    print(f"Temperature: {temperature} ({'greedy' if temperature == 0 else 'sampling'})")
    print("-" * 60)

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        item = dataset.data[idx]
        initial_board = item['initial_board']
        solution = item.get('solution', None)

        # Build solution from trajectory if not provided
        if solution is None:
            sol_board = [int(c) for c in initial_board]
            for pos, val in item['steps']:
                sol_board[pos] = val
            solution = ''.join(map(str, sol_board))

        # Count empty cells
        empty_cells = initial_board.count('0')
        empty_cells_list.append(empty_cells)

        # Build initial tokens from givens
        initial_tokens = build_initial_tokens(initial_board).unsqueeze(0).to(DEVICE)

        # Generate solution
        # Max new tokens = 2 * empty_cells (pos + val for each)
        max_new = empty_cells * 2

        with torch.no_grad():
            if temperature == 0:
                # Greedy decoding
                generated = initial_tokens.clone()
                for _ in range(max_new):
                    logits = model(generated)
                    next_logits = logits[:, -1, :]

                    # Constrain output
                    last_token = generated[0, -1].item()
                    mask = torch.full_like(next_logits, float('-inf'))
                    if last_token == SOS_TOKEN or is_val_token(last_token):
                        mask[0, 1:82] = 0  # Position tokens
                    elif is_pos_token(last_token):
                        mask[0, 82:91] = 0  # Value tokens
                    next_logits = next_logits + mask

                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
            else:
                generated = model.generate(
                    initial_tokens,
                    max_new_tokens=max_new,
                    temperature=temperature
                )

        # Decode to board
        tokens = generated[0].cpu().tolist()
        pred_board = decode_sequence_to_board(tokens, initial_board)
        sol_board = [int(c) for c in solution]

        # Compute metrics
        # Cell accuracy (only for originally empty cells)
        puzzle_correct = 0
        for i in range(81):
            if initial_board[i] == '0':  # Only count empty cells
                total_cells += 1
                if pred_board[i] == sol_board[i]:
                    correct_cells += 1
                    puzzle_correct += 1

        correct_per_puzzle.append(puzzle_correct / empty_cells if empty_cells > 0 else 1.0)

        # Solve rate
        total += 1
        if pred_board == sol_board and is_valid_sudoku(pred_board):
            solved += 1

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    solve_rate = solved / total * 100
    cell_acc = correct_cells / total_cells * 100 if total_cells > 0 else 0

    print(f"\n[Primary Metric]")
    print(f"  Solve Rate: {solved}/{total} = {solve_rate:.2f}%")

    print(f"\n[Secondary Metrics]")
    print(f"  Cell Accuracy (empty cells): {correct_cells}/{total_cells} = {cell_acc:.2f}%")
    print(f"  Avg cells correct per puzzle: {np.mean(correct_per_puzzle)*100:.2f}%")

    print(f"\n[Dataset Statistics]")
    print(f"  Avg empty cells: {np.mean(empty_cells_list):.1f}")
    print(f"  Min empty cells: {np.min(empty_cells_list)}")
    print(f"  Max empty cells: {np.max(empty_cells_list)}")

    print("=" * 60)

    return {
        'solve_rate': solve_rate,
        'cell_accuracy': cell_acc,
        'solved': solved,
        'total': total
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Sudoku AR Model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/sudoku-trajectory",
                        help="Path to dataset directory")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (train/test)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (None for all)")
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
