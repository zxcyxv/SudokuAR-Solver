import os
import numpy as np
import sys
from tqdm import tqdm

# Add correct path for norvig_solver import if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# We need to reuse the solver logic or just simple validation logic.
# Norvig solver's assign/eliminate logic is needed to verify if steps are valid.
# But actually, best verification is:
# 1. Start with initial board. 
# 2. Apply steps. 
# 3. Check if final board is valid and complete.

digits   = '123456789'
rows     = 'ABCDEFGHI'
cols     = digits
def cross(A, B): return [a + b for a in A for b in B]
squares  = cross(rows, cols)

def verify_sudoku(board_str):
    # Retrieve values
    values = {}
    for s, v in zip(squares, board_str):
        if v in digits:
            values[s] = v
        else:
            return False # Unfilled
    
    # Check constraints
    unitlist = ([cross(rows, c) for c in cols] +
                [cross(r, cols) for r in rows] +
                [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')])
    units = dict((s, [u for u in unitlist if s in u]) for s in squares)
    
    for s in squares:
        if s not in values: return False
        v = values[s]
        # Check all units
        for u in units[s]:
            # All values in unit must be unique
            unit_values = [values[s2] for s2 in u if s2 in values]
            if len(unit_values) != len(set(unit_values)):
                return False
    return True

def verify_dataset(data_path, num_samples=100):
    print(f"Verifying {data_path}...")
    try:
        data = np.load(data_path, allow_pickle=True)
    except FileNotFoundError:
        print("File not found.")
        return

    print(f"Total items: {len(data)}")
    
    if num_samples is None or num_samples > len(data):
        samples = data
    else:
        # random sample
        indices = np.random.choice(len(data), size=num_samples, replace=False)
        samples = data[indices]

    passed = 0
    failed = 0
    
    for item in tqdm(samples, desc="Verifying"):
        initial = item['initial_board']
        steps = item['steps']
        
        # Reconstruct
        board_list = list(initial)
        # Apply steps
        valid_steps = True
        for idx, val in steps:
            # Check if initially empty
            if initial[idx] != '0':
                print(f"Fail: Step writes to non-empty {idx}")
                valid_steps = False
                break
            
            # Write
            board_list[idx] = str(val)
        
        if not valid_steps:
            failed += 1
            continue
            
        final_board = "".join(board_list)
        
        # Check if full
        if '0' in final_board:
            print("Fail: Incomplete board")
            failed += 1
            continue
            
        # Check validity
        if verify_sudoku(final_board):
            passed += 1
        else:
            print(f"Fail: Invalid Sudoku\n{final_board}")
            failed += 1

    print(f"Verification Result: {passed}/{len(samples)} Passed")
    return passed == len(samples)

import glob

if __name__ == "__main__":
    base_dir = "data/sudoku-trajectory"
    
    # Train
    train_files = glob.glob(os.path.join(base_dir, "train_trajectories_part_*.npy"))
    if not train_files:
         train_files = [os.path.join(base_dir, "train_trajectories.npy")]
         
    for f in train_files:
        if os.path.exists(f):
            verify_dataset(f, num_samples=100) # Verify small sample of each chunk
            
    # Test
    test_files = glob.glob(os.path.join(base_dir, "test_trajectories_part_*.npy"))
    if not test_files:
         test_files = [os.path.join(base_dir, "test_trajectories.npy")]
         
    for f in test_files:
        if os.path.exists(f):
            verify_dataset(f, num_samples=100)
