"""
SudokuTransformer V2: Fixed Position Order + Value-Only Prediction

Key Changes from V1:
- Position is FIXED (raster scan order of empty cells)
- Model predicts VALUE only
- Simpler token vocabulary: [SOS, 1-9] = 10 tokens
- Board position passed via separate embedding

This simplifies the task:
  V1: P(Position, Value | context) - harder
  V2: P(Value | Position, context) - easier, Position is given
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
# Token Vocabulary (Simplified)
# -----------------------------------------------------------------------------
# Token layout (total 10 tokens):
#   0: [SOS] - Start of sequence
#   1-9: Values (digit 1-9)
#
# Position is NOT predicted - it's given via board_pos embedding

SOS_TOKEN = 0
VOCAB_SIZE = 10  # SOS + 9 values

def val_to_token(val):
    """Convert value (1-9) to token ID (1-9)"""
    return val

def token_to_val(token_id):
    """Convert token ID (1-9) to value (1-9)"""
    return token_id


# -----------------------------------------------------------------------------
# Model V2 (Fixed Position Order)
# -----------------------------------------------------------------------------
class SudokuTransformerV2(nn.Module):
    """
    Sudoku Transformer with Fixed Position Order.

    Key Design:
    -----------
    - Position is FIXED in raster scan order of empty cells
    - Model only predicts VALUE (1-9)
    - Board position info passed via separate embedding

    Input:
    ------
    - value_ids: [batch, seq_len] - Value tokens (0=SOS, 1-9=values)
    - board_pos: [batch, seq_len] - Board positions (0-80) for each token

    The model learns: P(Value_t | Value_{1:t-1}, BoardPos_{1:t})
    """

    def __init__(self,
                 num_layers=4,
                 hidden_dim=384,
                 num_heads=6,
                 max_seq_len=82,  # SOS + max 81 empty cells
                 vocab_size=VOCAB_SIZE,
                 board_size=81):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Value token embedding (SOS + 9 values)
        self.value_emb = nn.Embedding(vocab_size, hidden_dim)

        # Board position embedding (0-80)
        self.board_pos_emb = nn.Embedding(board_size, hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            DecoderBlock(hidden_dim, num_heads, hidden_dim * 4, max_seq_len)
            for _ in range(num_layers)
        ])

        self.norm_f = RMSNorm(hidden_dim)

        # Output head: predict next value (10 classes)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Weight tying
        self.value_emb.weight = self.lm_head.weight

    def forward(self, value_ids, board_pos):
        """
        Args:
            value_ids: [batch, seq_len] - Value token IDs
            board_pos: [batch, seq_len] - Board position for each token (0-80)

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Combine value and position embeddings
        x = self.value_emb(value_ids) + self.board_pos_emb(board_pos)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(self, initial_values, initial_pos, remaining_pos, temperature=1.0):
        """
        Autoregressive generation.

        Args:
            initial_values: [batch, n_givens+1] - SOS + given values
            initial_pos: [batch, n_givens+1] - Positions for initial values
            remaining_pos: [batch, n_empty] - Positions to fill (in order)
            temperature: Sampling temperature

        Returns:
            generated_values: [batch, n_empty] - Predicted values for empty cells
        """
        self.eval()
        device = initial_values.device
        batch_size = initial_values.shape[0]
        n_empty = remaining_pos.shape[1]

        # Start with initial sequence
        values = initial_values.clone()
        positions = initial_pos.clone()

        generated = []

        for i in range(n_empty):
            # Get next position to fill
            next_pos = remaining_pos[:, i:i+1]  # [batch, 1]

            # Forward pass
            logits = self.forward(values, positions)  # [batch, seq_len, vocab]
            next_logits = logits[:, -1, :]  # [batch, vocab]

            # Mask out SOS token (can't predict SOS)
            next_logits[:, SOS_TOKEN] = float('-inf')

            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Sample or argmax
            if temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token)

            # Append to sequence
            values = torch.cat([values, next_token], dim=1)
            positions = torch.cat([positions, next_pos], dim=1)

        return torch.cat(generated, dim=1)  # [batch, n_empty]
