import collections

# Peter Norvig's Sudoku Solver with Oracle Guide Trajectory Tracking

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a + b for a in A for b in B]

digits   = '123456789'
rows     = 'ABCDEFGHI'
cols     = digits
squares  = cross(rows, cols)
unitlist = ([cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')])
units = dict((s, [u for u in unitlist if s in u]) for s in squares)
peers = dict((s, set(sum(units[s], [])) - set([s])) for s in squares)

def parse_grid(grid, solution_grid_str=None):
    """Convert grid to a dict of possible values, {square: digits}, or
    return False if a contradiction is detected.
    
    If solution_grid_str is provided, it is used as the Oracle Guide.
    Returns: (values, trajectory)
    """
    # To start, every square can be any digit; then assign values from the grid.
    values = dict((s, digits) for s in squares)
    
    # We maintain a trajectory list. 
    # However, for the initial parsing/setup, we typically don't count givens as part of the "trajectory" 
    # unless requested. The user requested: "Output: trajectory list (Givens excluded, purely solved order)".
    # So we will track assignments, but we might filter givens later or just not record them if we know they are givens.
    # Actually, the user said: "Input: problem (grid) and answer (solution) both... Return: trajectory list (Givens excluded)"
    
    # Let's create a class or closure to manage state including the trajectory.
    solver = OracleSolver(grid, solution_grid_str)
    return solver.solve()

class OracleSolver:
    def __init__(self, initial_grid_str, solution_grid_str):
        self.initial_grid_str = initial_grid_str
        self.solution_grid_str = solution_grid_str
        self.values = dict((s, digits) for s in squares)
        self.trajectory = [] # List of (index, value) tuples
        self.givens = set()  # Set of square keys that were given

        # Pre-process solution grid into a simple map for Oracle
        # We need to correctly align squares with the input string.
        # Assuming input string order matches 'squares' list order (A1, A2, ..., I9)
        # Verify: cross('ABC', '123') -> ['A1', 'A2', 'A3', 'B1', ...] which is standard reading order.
        self.solution_map = {}
        if solution_grid_str:
            # Clean up string
            vals = [c for c in solution_grid_str if c in digits or c in '0.']
            if len(vals) == 81:
                for s, v in zip(squares, vals):
                    if v in digits:
                        self.solution_map[s] = v

    def solve(self):
        # 1. Parse Initial Grid
        # "0" or "." are empties.
        grid_values = [c for c in self.initial_grid_str if c in digits or c in '0.']
        if len(grid_values) != 81:
            # Handle potential different formats (ignore non-digit/non-dot chars)
             # Logic from Norvig's grid_values() often handles generic separators. 
             # Here we assume a relatively clean string or standard 81 chars.
             pass

        # Assign Givens
        # First, identify all givens to prevent recording them if solved by propagation
        for s, v in zip(squares, grid_values):
             if v in digits and v != '0':
                 self.givens.add(s)

        # Now apply assignments
        for s, v in zip(squares, grid_values):
            if v in digits and v != '0':
                if not self.assign(s, v, is_given=True):
                     return False, None # Contradiction in givens?

        # 2. Main Loop (Oracle Guide)
        # Repeat until all squares are solved (len(values[s]) == 1 for all s)
        while True:
            # Check if solved
            if all(len(self.values[s]) == 1 for s in squares):
                return self.values, self.trajectory
            
            # Find unsolved cell with min candidates
            # (s, n) where n is number of candidates, s is square
            unsolved = [(len(self.values[s]), s) for s in squares if len(self.values[s]) > 1]
            if not unsolved:
                # Should be caught by 'all solved' check, but if failure/contradiction happened (empty list), break
                return False, self.trajectory # Failure
            
            # Sort by number of candidates (entropy)
            n, s = min(unsolved)
            
            # Oracle Answer
            if s in self.solution_map:
                correct_digit = self.solution_map[s]
            else:
                # If no solution map provided or something wrong, we can't use Oracle.
                # Fallback or Error? Plan says "Input: problem and solution...". 
                # So we assume solution is available.
                return False, self.trajectory 
            
            # Assign
            # This is the key "Guide" step. We pick the correct digit.
            # Record this step in trajectory since it's an action by the solver on a non-given.
            if not self.assign(s, correct_digit, is_given=False):
                return False, self.trajectory # Contradiction with Oracle? Should not happen if valid.

    def assign(self, s, d, is_given=False):
        """Eliminate all the other values (except d) from values[s] and propagate.
        Return values, except return False if a contradiction is detected."""
        other_values = self.values[s].replace(d, '')
        if all(self.eliminate(s, d2) for d2 in other_values):
            # Success
            # If this assignment finalized the value (which it does, since we keep only d), 
            # and it's not a given, we add to trajectory?
            # Wait, `assign` calls `eliminate`. `eliminate` might reduce other squares to 1 value.
            # We want to capture the logical order.
            # Actually, standard Norvig behavior: `assign` calls `eliminate` for others.
            # `eliminate` checks if a square is reduced to 1 value. If so, it calls `eliminate` on peers.
            # We should probably hook into the moment a square becomes 'solved' (len=1).
            
            # BUT, the user plan says:
            # "Loop: ... Find min entropy ... Assign correct val ... Tracking: assign called -> add to list"
            # It also says: "assign이 호출되어 값이 확정될 때마다 trajectory 리스트에 (index, value)를 순서대로 기록합니다."
            # AND "제약 전파 과정에서 연쇄적으로 채워지는 칸들의 순서가 중요함".
            
            # So if `assign(s, d)` triggers `eliminate(peer, x)` which results in `peer` having only `y`,
            # that is also an "assignment" in a way (forced move).
            # The "Oracle Step" is just the *choice* when we are stuck (or just picking the next step).
            # The *Constraint Propagation* fills in the forced ones.
            
            # So, the trajectory should be recorded whenever `len(self.values[s])` becomes 1.
            # We can do this in `eliminate`.
            
            # Re-reading user request: 
            # "assign 함수가 성공적으로 값을 확정할 때(후보가 1개만 남거나, 정답을 대입했을 때), 그 (cell_index, value)를 리스트에 추가합니다."
            # "중요: 이미 trajectory에 들어있는 칸은 중복해서 넣지 않도록 체크해야 합니다."
            
            return self.values
        else:
            return False

    def eliminate(self, s, d):
        """Eliminate d from values[s]; propagate when values or places <= 2.
        Return values, except return False if a contradiction is detected."""
        if d not in self.values[s]:
            return self.values ## Already eliminated
        
        self.values[s] = self.values[s].replace(d, '')
        
        # (1) If a square s is reduced to one value d2, then eliminate d2 from the peers.
        if len(self.values[s]) == 0:
            return False ## Contradiction: removed last value
        elif len(self.values[s]) == 1:
            d2 = self.values[s]
            
            # TRACKING: Value confirmed.
            self.record_step(s, d2)
            
            if not all(self.eliminate(s2, d2) for s2 in peers[s]):
                return False
        
        # (2) If a unit u is reduced to only one place for a value d, then put it there.
        for u in units[s]:
            dplaces = [s for s in u if d in self.values[s]]
            if len(dplaces) == 0:
                return False ## Contradiction: no place for this value
            elif len(dplaces) == 1:
                # d can only be in one place in unit; assign it there
                if not self.assign(dplaces[0], d):
                    return False
        return self.values

    def record_step(self, s, d):
        # Only record if not given and not already recorded
        if s not in self.givens:
            # We need to know if we already recorded this one.
            # We can use a set for checking 'recorded' state efficiently, 
            # but trajectory list needs order.
            # Let's add a `recorded` set to the class.
            if not hasattr(self, 'recorded'):
                self.recorded = set()
            
            if s not in self.recorded:
                self.recorded.add(s)
                # Convert 'A1'.. to 0..80 index
                idx = squares.index(s) 
                # Convert '1'..'9' to integer? Or string?
                # User example: (idx, val) -> (0, 5)
                val = int(d)
                self.trajectory.append((idx, val))

def solve_with_trajectory(initial_grid_str, solution_grid_str):
    """
    Solves the Sudoku using Oracle Guide logic and returns the trajectory.
    
    Args:
        initial_grid_str: String representation of the unsolved puzzle (81 chars).
        solution_grid_str: String representation of the solved puzzle (81 chars).
        
    Returns:
        trajectory: List of (index, value) tuples for non-given cells, in solved order.
    """
    solver = OracleSolver(initial_grid_str, solution_grid_str)
    values, trajectory = solver.solve()
    # If failed, trajectory might be partial or empty? 
    # With Oracle it shouldn't fail unless the puzzle/solution is mismatched.
    return trajectory
