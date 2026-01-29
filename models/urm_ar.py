
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class URMConfig:
    vocab_size: int = 730
    hidden_size: int = 384
    num_layers: int = 4
    num_heads: int = 6
    expansion: float = 4.0
    n_recurrence: int = 8  # Default to 8 (Inner Loop count from paper)
    max_seq_len: int = 82
    dropout: float = 0.0
    rope_theta: float = 10000.0

# -----------------------------------------------------------------------------
# RoPE (Copied & Adapted)
# -----------------------------------------------------------------------------
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [b, s, h, d]
    # cos, sin: [1, s, 1, d]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # return [1, seq_len, 1, dim] for broadcasting
        return emb.cos().unsqueeze(0).unsqueeze(2), emb.sin().unsqueeze(0).unsqueeze(2)

# -----------------------------------------------------------------------------
# Layers
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class CausalAttention(nn.Module):
    def __init__(self, config: URMConfig):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0
        self.head_dim = config.hidden_size // config.num_heads
        self.num_heads = config.num_heads
        
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x):
        b, s, d = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(b, s, self.num_heads, self.head_dim)
        k = k.view(b, s, self.num_heads, self.head_dim)
        v = v.view(b, s, self.num_heads, self.head_dim)

        cos, sin = self.rope(q, seq_len=s)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose for SDPA: [b, h, s, d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Causal Mask is handled efficiently by is_causal=True in scaled_dot_product_attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.o_proj(out)

class ConvSwiGLU_AR(nn.Module):
    """
    Autoregressive (Causal) ConvSwiGLU.
    CRITICAL: Uses explicit left-padding to prevent information leakage.
    """
    def __init__(self, config: URMConfig):
        super().__init__()
        hidden = config.hidden_size
        inter = int(config.expansion * hidden * 2 / 3) # Approximate expansion logic
        # Ensure multiple of 256 for efficiency (optional, matching style)
        inter = (inter + 255) // 256 * 256
        
        self.gate_up_proj = nn.Linear(hidden, inter * 2, bias=False)
        
        # Kernel size 2:  Output[t] depends on Input[t] and Input[t-1]
        self.kernel_size = 2
        self.dwconv = nn.Conv1d(
            in_channels=inter,
            out_channels=inter,
            kernel_size=self.kernel_size,
            groups=inter,
            bias=True,
            padding=0 # We will do manual padding
        )
        
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: [b, s, d]
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        
        # Mixing: SiLU(gate) * up
        # NOTE: URM Paper Figure 2 says Conv is inside gate branch, but Eq 136 says Conv is after mixing.
        # User feedback and Equ 136 confirm: Conv is applied to the mixed output.
        x_ffn = self.act(gate) * up  # [b, s, inter]
        
        # CAUSAL CONVOLUTION IMPLEMENTATION
        # Transpose to [b, inter, s] for Conv1d
        x_conv_in = x_ffn.transpose(1, 2)
        
        # Left Pad: (kernel_size - 1) zeros on the left
        # Padding applied to last dim (time)
        x_conv_in = F.pad(x_conv_in, (self.kernel_size - 1, 0))
        
        out_conv = self.dwconv(x_conv_in)
        # Output length should match input length (s)
        
        out_conv = self.act(out_conv)
        out_conv = out_conv.transpose(1, 2) # Back to [b, s, inter]
        
        return self.down_proj(out_conv)

class URMBlock(nn.Module):
    def __init__(self, config: URMConfig):
        super().__init__()
        self.rms_norm_eps = 1e-6
        self.self_attn = CausalAttention(config)
        self.mlp = ConvSwiGLU_AR(config) # "Short Convolution"
        
    def forward(self, x):
        # Pre-Norm architecture
        x = x + self.self_attn(RMSNorm(x.shape[-1], self.rms_norm_eps)(x))
        x = x + self.mlp(RMSNorm(x.shape[-1], self.rms_norm_eps)(x))
        return x

class SudokuURM_AR(nn.Module):
    def __init__(self, config: Optional[URMConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            # Allow kwargs override for flexible init similar to previous model
            config = URMConfig(**kwargs)
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Loop Coordinate Embedding: Learnable embedding for each recurrence step
        # +1 for safety or 0-indexing
        self.loop_emb = nn.Embedding(config.n_recurrence, config.hidden_size)
        
        # URM uses a *single* stack of layers (Transition Function) repeatedly
        self.layers = nn.ModuleList([
            URMBlock(config) for _ in range(config.num_layers)
        ])
        
        self.norm_f = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        self.token_emb.weight = self.lm_head.weight

    def forward(self, token_ids):
        # token_ids: [b, s]
        x = self.token_emb(token_ids)
        b, s, d = x.shape
        
        # TBPTL: Freeze gradients for the first `n_freeze` steps of recurrence
        # This matches the paper's "forward-only" training for initial loops
        n_freeze = 2 

        # Recurrence Loop (Effective Depth = num_layers * n_recurrence)
        for step in range(self.config.n_recurrence):
            # TBPTL Logic: Detach state to stop gradient back-flow for early steps
            if step < n_freeze:
                x = x.detach()

            # Add Loop Coordinate Embedding
            # We broadcast step_emb [1, 1, d] to [b, s, d]
            step_idx = torch.tensor([step], device=x.device)
            loop_signal = self.loop_emb(step_idx).unsqueeze(1) # [1, 1, d]
            
            # Inject loop signal into hidden state
            x = x + loop_signal
            
            # Apply Transition Function (Shared Layers)
            for layer in self.layers:
                x = layer(x)
                
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, initial_board: str, max_actions: int = 81,
                 temperature: float = 0.0, device: str = 'cpu'):
        """
        Stateless Generation for Correctness.
        Re-forwards the entire sequence at each step.
        """
        from models.transformer_v3 import SOS_TOKEN, NUM_ACTIONS, action_to_token, token_to_action, is_action_token
        
        self.eval()
        board = [int(c) for c in initial_board]
        filled = set(i for i, v in enumerate(board) if v != 0)

        given_tokens = []
        for cell_id, val in enumerate(board):
            if val != 0:
                token = action_to_token(cell_id, val)
                given_tokens.append(token)

        start_tokens = [SOS_TOKEN] + given_tokens
        tokens = torch.tensor([start_tokens], device=device) # [1, seq_len]
        actions = []

        valid_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool, device=device)
        for cell_id in range(81):
            if cell_id not in filled:
                start_tok = cell_id * 9
                valid_mask[start_tok:start_tok + 9] = True

        for _ in range(max_actions):
            # Forward (Full Sequence)
            logits = self.forward(tokens)
            next_logits = logits[0, -1, :NUM_ACTIONS]

            next_logits = next_logits.masked_fill(~valid_mask, float('-inf'))

            if temperature == 0:
                next_token = next_logits.argmax().unsqueeze(0).unsqueeze(0)
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).unsqueeze(0)

            token_val = next_token.item()
            if not is_action_token(token_val):
                break

            cell_id, value = token_to_action(token_val)
            actions.append((cell_id, value))
            board[cell_id] = value
            filled.add(cell_id)

            start_tok = cell_id * 9
            valid_mask[start_tok:start_tok + 9] = False

            tokens = torch.cat([tokens, next_token], dim=1)

            if len(filled) == 81:
                break

        return actions, board
