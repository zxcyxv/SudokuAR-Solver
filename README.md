# Sudoku Autoregressive Solver (AR-TRM)

This project implements an Autoregressive Transformer to solve Sudoku puzzles by learning the optimal logical trajectory.

## 1. Setup Environment
Clone the repository and install dependencies using `uv` (or pip).

```bash
# Initialize and install dependencies
uv sync
```

## 2. Generate Dataset (One-time)
Download the Sudoku dataset and generate the solution trajectories. This process uses the Norvig Oracle Guide to create optimal paths.

```bash
# Generates train/test trajectories in dataset/data/sudoku-trajectory
# Uses multiprocessing for speed.
uv run python dataset/build_trajectory_dataset.py
```
*Note: This generates ~1.7GB of data.*

## 3. Verify Dataset (Optional)
Check the integrity of the generated data.

```bash
uv run python dataset/verify_dataset.py
```

## 4. Train Model
Train the AR Transformer model.

```bash
# Trains the model and saves checkpoints to /checkpoints
uv run python train.py
```

## 5. Visualization (Optional)
Visualize the solving process of the generated trajectories.

```bash
uv run python dataset/visualize_trajectory.py
```
