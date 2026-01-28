import pygame
import numpy as np
import os
import sys

# Parameters
WIDTH, HEIGHT = 600, 650
GRID_SIZE = 540
CELL_SIZE = GRID_SIZE // 9
OFFSET_X = (WIDTH - GRID_SIZE) // 2
OFFSET_Y = 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_BLUE = (173, 216, 230)
GIVEN_COLOR = (0, 0, 0)
STEP_COLOR = (0, 0, 255)
HIGHLIGHT_COLOR = (255, 255, 150)

class SudokuVisualizer:
    def __init__(self, data_path):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Sudoku Trajectory Visualizer")
        self.font = pygame.font.SysFont("Arial", 36)
        self.small_font = pygame.font.SysFont("Arial", 20)
        
        # Load Data
        if not os.path.exists(data_path):
            print(f"Error: {data_path} not found.")
            sys.exit()
            
        self.data = np.load(data_path, allow_pickle=True)
        self.current_puzzle_idx = 0
        self.reset_puzzle()

    def reset_puzzle(self):
        puzzle_data = self.data[self.current_puzzle_idx]
        self.initial_board = puzzle_data['initial_board']
        self.steps = puzzle_data['steps']
        self.rating = puzzle_data.get('rating', 'N/A')
        
        self.current_board = list(self.initial_board)
        self.current_step_idx = 0
        self.last_changed_cell = -1

    def draw_grid(self):
        self.screen.fill(WHITE)
        
        # Draw cells and values
        for i in range(81):
            row, col = divmod(i, 9)
            x = OFFSET_X + col * CELL_SIZE
            y = OFFSET_Y + row * CELL_SIZE
            
            # Highlight most recent step
            if i == self.last_changed_cell:
                pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, (x, y, CELL_SIZE, CELL_SIZE))
            
            val = self.current_board[i]
            if val != '0':
                color = GIVEN_COLOR if self.initial_board[i] != '0' else STEP_COLOR
                text = self.font.render(val, True, color)
                self.screen.blit(text, (x + CELL_SIZE//3, y + CELL_SIZE//6))

        # Draw lines
        for i in range(10):
            line_width = 3 if i % 3 == 0 else 1
            # Vertical
            pygame.draw.line(self.screen, BLACK, 
                             (OFFSET_X + i * CELL_SIZE, OFFSET_Y), 
                             (OFFSET_X + i * CELL_SIZE, OFFSET_Y + GRID_SIZE), line_width)
            # Horizontal
            pygame.draw.line(self.screen, BLACK, 
                             (OFFSET_X, OFFSET_Y + i * CELL_SIZE), 
                             (OFFSET_X + GRID_SIZE, OFFSET_Y + i * CELL_SIZE), line_width)

        # Draw UI Info
        info_text = f"Puzzle: {self.current_puzzle_idx + 1}/{len(self.data)} | Rating: {self.rating} | Step: {self.current_step_idx}/{len(self.steps)}"
        info_surface = self.small_font.render(info_text, True, BLACK)
        self.screen.blit(info_surface, (OFFSET_X, HEIGHT - 70))
        
        help_text = "[Click/Space]: Next Step | [N]: Next Puzzle | [R]: Reset | [Esc]: Quit"
        help_surface = self.small_font.render(help_text, True, BLACK)
        self.screen.blit(help_surface, (OFFSET_X, HEIGHT - 40))

    def next_step(self):
        if self.current_step_idx < len(self.steps):
            cell_idx, val = self.steps[self.current_step_idx]
            self.current_board[cell_idx] = str(val)
            self.last_changed_cell = cell_idx
            self.current_step_idx += 1
            return True
        return False

    def run(self):
        running = True
        while running:
            self.draw_grid()
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in [pygame.K_SPACE, pygame.K_RETURN, pygame.K_RIGHT]:
                        self.next_step()
                    elif event.key == pygame.K_n:
                        self.current_puzzle_idx = (self.current_puzzle_idx + 1) % len(self.data)
                        self.reset_puzzle()
                    elif event.key == pygame.K_r:
                        self.reset_puzzle()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Left click
                        self.next_step()
        
        pygame.quit()

if __name__ == "__main__":
    # Defaulting to train set for visualization
    data_path = os.path.join("data", "sudoku-trajectory", "train_trajectories.npy")
    if not os.path.exists(data_path):
        # Try local folder if run from dataset dir
        data_path = os.path.join("..", "data", "sudoku-trajectory", "train_trajectories.npy")
        
    visualizer = SudokuVisualizer(data_path)
    visualizer.run()
