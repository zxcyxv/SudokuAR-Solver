from typing import Optional, List, Dict, Any
import os
import csv
import json
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from argdantic import ArgParser
from pydantic import BaseModel

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from norvig_solver import solve_with_trajectory

cli = ArgParser()

class TrajectoryDataConfig(BaseModel):
    source_repo: str = "sapientinc/sudoku-extreme"
    output_dir: str = "data/sudoku-trajectory"
    max_samples: Optional[int] = None # For testing/limiting
    min_rating: int = 0

import multiprocessing

def process_row(args):
    """
    Worker function to process a single row.
    args: (source, puzzle, solution, rating, min_rating)
    Returns: dict or None
    """
    source, puzzle, solution, rating, min_rating = args
    
    if int(rating) < min_rating:
        return None
    
    # Verify length
    if len(puzzle) != 81 or len(solution) != 81:
        return None
    
    # Check for clean data (dots or zeros)
    puzzle_clean = puzzle.replace('.', '0')
    solution_clean = solution.replace('.', '0')
    
    try:
        steps = solve_with_trajectory(puzzle_clean, solution_clean)
        
        # Store result
        item = {
            "initial_board": puzzle_clean,
            "steps": steps, # List of (index, value)
            "rating": int(rating)
        }
        return item
        
    except Exception as e:
        # print(f"Error processing puzzle: {e}")
        return None

@cli.command(singleton=True)
def generate_trajectories(config: TrajectoryDataConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Determine number of processes
    num_processes = max(1, multiprocessing.cpu_count() - 2) # Leave some cores for system
    print(f"Using {num_processes} processes for generation.")
    
    for split in ["train", "test"]:
        print(f"Processing {split} set...")
        
        # Download/Load CSV
        csv_path = hf_hub_download(config.source_repo, f"{split}.csv", repo_type="dataset")
        
        with open(csv_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader) # source, puzzle, solution, rating
            
            rows = list(reader)
            if config.max_samples:
                 rows = rows[:config.max_samples]
            
            # Prepare arguments for workers
            # We pass tuples to keep pickling overhead low
            tasks = [
                (r[0], r[1], r[2], r[3], config.min_rating) 
                for r in rows
            ]
            
            chunk_size = 50000
            chunk_index = 0
            processed_data = []
            
            # Use Pool
            with multiprocessing.Pool(processes=num_processes) as pool:
                # imap_unordered is faster if order doesn't matter (it typically doesn't for training data)
                # chunksize in imap helps reduce IPC overhead
                iterator = pool.imap_unordered(process_row, tasks, chunksize=100)
                
                for result in tqdm(iterator, total=len(tasks), desc=split):
                    if result is not None:
                        processed_data.append(result)
                        
                        if len(processed_data) >= chunk_size:
                            output_file = os.path.join(config.output_dir, f"{split}_trajectories_part_{chunk_index}.npy")
                            np.save(output_file, processed_data)
                            processed_data = []
                            chunk_index += 1
            
            # Save remaining
            if processed_data:
                output_file = os.path.join(config.output_dir, f"{split}_trajectories_part_{chunk_index}.npy")
                print(f"Saving final chunk {chunk_index} ({len(processed_data)} items) to {output_file}...")
                np.save(output_file, processed_data)

if __name__ == "__main__":
    # Support for Windows multiprocessing
    multiprocessing.freeze_support()
    cli()
