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
        # Precompute theta
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len=None):
        # x: [b, s, n_head, head_dim] (usually)
        # But we need t corresponding to seq len
        if seq_len is None:
            seq_len = x.shape[1]

        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different implementations exist. Common is [cos, sin]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [b, s, h, d]
    # cos, sin: [s, d] -> [1, s, 1, d] to broadcast
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
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False) # Gate
        self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False) # Val
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False) # Out

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

        # RoPE
        cos, sin = self.rope(q, seq_len=s)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Flash Attention is cleaner if available, but manual for clarity/compatibility
        # Transpose for [b, h, s, d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled Dot Product
        # Causal Mask
        # torch.nn.functional.scaled_dot_product_attention handles is_causal=True efficiently
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
# Token Vocabulary for Serialized Sequence
# -----------------------------------------------------------------------------
# Token layout (total 91 tokens):
#   0: [SOS] - Start of sequence
#   1-81: Position tokens (board cell index 0-80)
#   82-90: Value tokens (digit 1-9)
#
# This design follows the Chain Rule of Probability:
#   P(C_{t+1}, V_{t+1} | S_{1:t}) = P(C_{t+1} | S_{1:t}) × P(V_{t+1} | C_{t+1}, S_{1:t})
#
# The serialized sequence [SOS, P0, V0, P1, V1, ...] ensures that:
#   - When predicting V_t, the model has already seen C_t in its context
#   - This eliminates the conditional independence assumption error

SOS_TOKEN = 0
POS_TOKEN_OFFSET = 1      # Position 0 -> Token 1, Position 80 -> Token 81
VAL_TOKEN_OFFSET = 82     # Value 1 -> Token 82, Value 9 -> Token 90
VOCAB_SIZE = 91           # 1 (SOS) + 81 (positions) + 9 (values 1-9)

def pos_to_token(pos_idx):
    """Convert board position (0-80) to token ID (1-81)"""
    return pos_idx + POS_TOKEN_OFFSET

def val_to_token(val):
    """Convert value (1-9) to token ID (82-90)"""
    return val - 1 + VAL_TOKEN_OFFSET

def token_to_pos(token_id):
    """Convert token ID (1-81) to board position (0-80)"""
    return token_id - POS_TOKEN_OFFSET

def token_to_val(token_id):
    """Convert token ID (82-90) to value (1-9)"""
    return token_id - VAL_TOKEN_OFFSET + 1

def is_pos_token(token_id):
    """Check if token is a position token (1-81)"""
    return POS_TOKEN_OFFSET <= token_id <= 81

def is_val_token(token_id):
    """Check if token is a value token (82-90)"""
    return VAL_TOKEN_OFFSET <= token_id <= 90

# -----------------------------------------------------------------------------
# Model (Serialized Autoregressive)
# -----------------------------------------------------------------------------
class SudokuTransformer(nn.Module):
    """
    Serialized Autoregressive Transformer for Sudoku.

    Key Design Change (Fixing Independence Assumption):
    ---------------------------------------------------
    Previous (WRONG):
        P(C, V | S) ≈ P(C | S) × P(V | S)  [Parallel heads]

    Current (CORRECT):
        P(C, V | S) = P(C | S) × P(V | C, S)  [Serialized sequence]

    The model now processes a flattened sequence:
        [SOS, P_0, V_0, P_1, V_1, ..., P_n, V_n]

    When predicting V_i, the model has already observed P_i in its context,
    satisfying the chain rule of probability.

    Token Vocabulary:
        - Token 0: [SOS]
        - Tokens 1-81: Position indices (0-80)
        - Tokens 82-90: Values (1-9)
    """

    def __init__(self,
                 num_layers=2,
                 hidden_dim=384,
                 num_heads=6,
                 max_seq_len=163,  # 1 (SOS) + 81*2 (pos-val pairs) = 163
                 vocab_size=VOCAB_SIZE):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Unified token embedding
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            DecoderBlock(hidden_dim, num_heads, hidden_dim * 4, max_seq_len)
            for _ in range(num_layers)
        ])

        self.norm_f = RMSNorm(hidden_dim)

        # Single unified output head
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Weight tying (optional but common in LLMs)
        self.token_emb.weight = self.lm_head.weight

    def forward(self, token_ids):
        """
        Args:
            token_ids: [batch, seq_len] - Token IDs from unified vocabulary

        Returns:
            logits: [batch, seq_len, vocab_size] - Next token prediction logits
        """
        x = self.token_emb(token_ids)  # [b, s, hidden_dim]

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)

        logits = self.lm_head(x)  # [b, s, vocab_size]

        return logits

    @torch.no_grad()
    def generate(self, initial_tokens, max_new_tokens=162, temperature=1.0,
                 constrain_output=True):
        """
        Autoregressive generation with optional output constraints.

        Args:
            initial_tokens: [batch, seq_len] - Starting tokens (e.g., [SOS] + givens)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = no change)
            constrain_output: If True, mask invalid tokens based on position

        Returns:
            generated: [batch, seq_len + max_new_tokens] - Full sequence
        """
        self.eval()
        device = initial_tokens.device
        batch_size = initial_tokens.shape[0]

        generated = initial_tokens.clone()

        for step in range(max_new_tokens):
            # Get current sequence length
            seq_len = generated.shape[1]

            # Forward pass
            logits = self.forward(generated)  # [b, s, vocab_size]

            # Get logits for the last position
            next_logits = logits[:, -1, :]  # [b, vocab_size]

            if constrain_output:
                # Determine if we should predict position or value
                # After SOS or after a value token -> predict position
                # After a position token -> predict value
                last_token = generated[:, -1]  # [b]

                # Create mask
                mask = torch.full_like(next_logits, float('-inf'))

                for b in range(batch_size):
                    lt = last_token[b].item()
                    if lt == SOS_TOKEN or is_val_token(lt):
                        # Next should be a position token (1-81)
                        mask[b, POS_TOKEN_OFFSET:POS_TOKEN_OFFSET+81] = 0
                    elif is_pos_token(lt):
                        # Next should be a value token (82-90)
                        mask[b, VAL_TOKEN_OFFSET:VAL_TOKEN_OFFSET+9] = 0

                next_logits = next_logits + mask

            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Sample or argmax
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [b, 1]

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Early stopping if all sequences complete (81 moves = 162 tokens)
            # Each move is (pos, val), so 81 moves = 162 additional tokens after SOS
            if generated.shape[1] >= 163:
                break

        return generated


# Legacy compatibility wrapper (optional - for gradual migration)
class SudokuTransformerLegacy(nn.Module):
    """
    Legacy two-head model for reference/comparison.
    WARNING: This has the independence assumption error.
    """
    def __init__(self,
                 num_layers=2,
                 hidden_dim=384,
                 num_heads=6,
                 max_seq_len=81,
                 vocab_val_size=11,
                 vocab_pos_size=81):
        super().__init__()

        self.pos_emb = nn.Embedding(vocab_pos_size, hidden_dim)
        self.val_emb = nn.Embedding(vocab_val_size, hidden_dim)

        self.layers = nn.ModuleList([
            DecoderBlock(hidden_dim, num_heads, hidden_dim * 4, max_seq_len)
            for _ in range(num_layers)
        ])

        self.norm_f = RMSNorm(hidden_dim)

        self.pointer_head = nn.Linear(hidden_dim, 81, bias=False)
        self.value_head = nn.Linear(hidden_dim, 10, bias=False)

    def forward(self, pos_idx, val_idx):
        x = self.pos_emb(pos_idx) + self.val_emb(val_idx)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)

        logits_pos = self.pointer_head(x)
        logits_val = self.value_head(x)

        return logits_pos, logits_val
