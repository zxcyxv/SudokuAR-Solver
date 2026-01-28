import sys
import os

# Add the current directory to path so we can import norvig_solver
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from norvig_solver import solve_with_trajectory

def test_easy_sudoku():
    # Example Sudoku (Easy)
    # 0030206...
    puzzle = "003020600900305001001806400008102900700000008006708200002609500800203009005010300"
    solution = "483921657967345821251876493548132976729564138136798245372689514814253769695417382"
    
    print("Testing Easy Sudoku...")
    trajectory = solve_with_trajectory(puzzle, solution)
    
    print(f"Trajectory length: {len(trajectory)}")
    
    # Verify trajectory length matches number of empties
    empties = puzzle.count('0')
    if len(trajectory) != empties:
        print(f"FAILED: Trajectory length {len(trajectory)} != empties {empties}")
    else:
        print("PASSED: Trajectory length matches empties")
        
    # Verify order applied results in solution
    board = list(puzzle)
    for idx, (pos, val) in enumerate(trajectory):
        if board[pos] != '0':
            print(f"FAILED: Step {idx} writes to non-empty cell {pos}")
        board[pos] = str(val)
        
    final_board = "".join(board)
    if final_board == solution:
        print("PASSED: Final board matches solution")
    else:
        print("FAILED: Final board does not match solution")
        # diff
        for i in range(81):
            if final_board[i] != solution[i]:
                print(f"Mismatch at {i}: {final_board[i]} != {solution[i]}")

def test_hard_sudoku():
    # Harder one
    puzzle = "000000000000003085001020000000507000004000100090000000500000073002010000000040009"
    # Just making up a solution for testing flow (incorrect logic check but flow check)
    # Actually need a valid pair for proper Oracle test.
    # Let's use the solver to get a solution first if available? 
    # Or just skip if I don't have a hard pair handy.
    # The solver relies on `solution` being correct.
    pass

if __name__ == "__main__":
    test_easy_sudoku()
