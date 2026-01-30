
"""
SudokuTransformer V3: Unified Action Token (Position + Value Combined)

Token Design:
-------------
- Token = Cell_ID * 9 + (Value - 1)
- Cell_ID = Row * 9 + Col (0-80)
- Value = 1-9
- Token Range: 0 ~ 728 (729 tokens)
- Plus SOS token: 729
- Total Vocab: 730

Example:
- Token 0 = Cell(0,0) + Value 1
- Token 8 = Cell(0,0) + Value 9
- Token 9 = Cell(0,1) + Value 1
- Token 728 = Cell(8,8) + Value 9

Decoding:
- Cell_ID = Token // 9
- Value = (Token % 9) + 1

This approach:
1. Predicts "where + what" in a single token
2. Uses Oracle order (easiest cells first)
3. Standard next-token prediction like GPT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------------------------------------------------------------
# RoPE
# -----------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def is_valid_sudoku(board):
    """
    Check if a 81-length list represents a valid solved Sudoku board.
    board: List of 81 integers (1-9)
    """
    if len(board) != 81 or any(v < 1 or v > 9 for v in board):
        return False

    # Check rows
    for i in range(9):
        row = board[i*9 : (i+1)*9]
        if len(set(row)) != 9:
            return False

    # Check columns
    for i in range(9):
        col = [board[i + j*9] for j in range(9)]
        if len(set(col)) != 9:
            return False

    # Check 3x3 boxes
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            box = []
            for r in range(3):
                for c in range(3):
                    box.append(board[(i + r) * 9 + (j + c)])
            if len(set(box)) != 9:
                return False

    return True

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# -----------------------------------------------------------------------------
# Components
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, max_seq_len):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.n_head = n_head
        self.head_dim = dim // n_head

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x):
        b, s, d = x.shape

        q = self.wq(x).view(b, s, self.n_head, self.head_dim)
        k = self.wk(x).view(b, s, self.n_head, self.head_dim)
        v = self.wv(x).view(b, s, self.n_head, self.head_dim)

        cos, sin = self.rope(q, seq_len=s)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.wo(out)


class DecoderBlock(nn.Module):
    def __init__(self, dim, n_head, intermediate_dim, max_seq_len):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_head, max_seq_len)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, intermediate_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# -----------------------------------------------------------------------------
# Token Vocabulary
# -----------------------------------------------------------------------------
# Action tokens: 0 ~ 728 (Cell_ID * 9 + Value - 1)
# SOS token: 729
# Total: 730

NUM_ACTIONS = 729  # 81 cells * 9 values
SOS_TOKEN = 729
VOCAB_SIZE = 730


def action_to_token(cell_id: int, value: int) -> int:
    """Convert (cell_id, value) to token. cell_id: 0-80, value: 1-9"""
    return cell_id * 9 + (value - 1)


def token_to_action(token: int) -> tuple:
    """Convert token to (cell_id, value). Returns (cell_id, value)"""
    cell_id = token // 9
    value = (token % 9) + 1
    return cell_id, value


def is_action_token(token: int) -> bool:
    """Check if token is an action token (not SOS)"""
    return 0 <= token < NUM_ACTIONS


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class SudokuTransformerV3(nn.Module):
    """
    Sudoku Transformer with Unified Action Tokens.

    Each token represents a complete action: "Place value V at cell C"
    Token = Cell_ID * 9 + (Value - 1)

    Sequence format:
    [SOS, Action1, Action2, ..., ActionN]

    Where actions are in ORACLE ORDER (easiest cells first).
    """

    def __init__(self,
                 num_layers=4,
                 hidden_dim=384,
                 num_heads=6,
                 max_seq_len=82,  # SOS + max 81 actions
                 vocab_size=VOCAB_SIZE):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            DecoderBlock(hidden_dim, num_heads, hidden_dim * 4, max_seq_len)
            for _ in range(num_layers)
        ])

        self.norm_f = RMSNorm(hidden_dim)

        # Output head
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Weight tying
        self.token_emb.weight = self.lm_head.weight

    def forward(self, token_ids):
        """
        Args:
            token_ids: [batch, seq_len] - Token IDs

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        x = self.token_emb(token_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(self, initial_board: str, max_actions: int = 81,
                 temperature: float = 0.0, device: str = 'cpu'):
        """
        Generate solution for a puzzle.

        Args:
            initial_board: 81-char string (0 for empty)
            max_actions: Maximum actions to generate
            temperature: 0 for greedy, >0 for sampling
            device: Device to run on

        Returns:
            actions: List of (cell_id, value) tuples
            final_board: List of 81 integers
        """
        self.eval()

        # Parse initial board and create mask of filled cells
        board = [int(c) for c in initial_board]
        filled = set(i for i, v in enumerate(board) if v != 0)

        # Build Givens tokens (raster order: 0->80) - CRITICAL: Model needs to see initial board
        given_tokens = []
        for cell_id, val in enumerate(board):
            if val != 0:
                token = action_to_token(cell_id, val)
                given_tokens.append(token)

        # Start with [SOS, Given1, Given2, ..., GivenN]
        start_tokens = [SOS_TOKEN] + given_tokens
        tokens = torch.tensor([start_tokens], device=device)
        actions = []

        # Precompute valid token mask (vectorized) - updated each step
        valid_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool, device=device)
        for cell_id in range(81):
            if cell_id not in filled:
                start_tok = cell_id * 9
                valid_mask[start_tok:start_tok + 9] = True

        for _ in range(max_actions):
            # Forward
            logits = self.forward(tokens)
            next_logits = logits[0, -1, :NUM_ACTIONS]  # [729] - exclude SOS

            # Apply mask (invalid tokens get -inf)
            next_logits = next_logits.masked_fill(~valid_mask, float('-inf'))

            # Sample or argmax
            if temperature == 0:
                next_token = next_logits.argmax().unsqueeze(0).unsqueeze(0)
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).unsqueeze(0)

            # Decode action
            token_val = next_token.item()
            if not is_action_token(token_val):
                break

            cell_id, value = token_to_action(token_val)
            actions.append((cell_id, value))
            board[cell_id] = value
            filled.add(cell_id)

            # Update valid mask - mark this cell as filled (invalid for future)
            start_tok = cell_id * 9
            valid_mask[start_tok:start_tok + 9] = False

            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)

            # Check if board is complete
            if len(filled) == 81:
                break

        return actions, board
