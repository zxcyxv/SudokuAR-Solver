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
# Model
# -----------------------------------------------------------------------------
class SudokuTransformer(nn.Module):
    def __init__(self, 
                 num_layers=2, 
                 hidden_dim=384, 
                 num_heads=6, 
                 max_seq_len=81, # 81 cells max sequence
                 vocab_val_size=11, # 0..9 + pad? 1-9 are values. 0 is start/pad. Range 0-10->11?
                 vocab_pos_size=81):
        super().__init__()
        
        self.pos_emb = nn.Embedding(vocab_pos_size, hidden_dim) # Board Position (0-80)
        self.val_emb = nn.Embedding(vocab_val_size, hidden_dim) # Cell Value
        
        self.layers = nn.ModuleList([
            DecoderBlock(hidden_dim, num_heads, hidden_dim * 4, max_seq_len)
            for _ in range(num_layers)
        ])
        
        self.norm_f = RMSNorm(hidden_dim)
        
        # Heads
        self.pointer_head = nn.Linear(hidden_dim, 81, bias=False) # Next coordinate logits (0-80)
        self.value_head = nn.Linear(hidden_dim, 10, bias=False) # Next value logits (1-9). Using 10 to include 0 index safely or just mask it.

    def forward(self, pos_idx, val_idx):
        # pos_idx: [b, s] (0-80)
        # val_idx: [b, s] (0-9)
        
        # Combined Embedding
        x = self.pos_emb(pos_idx) + self.val_emb(val_idx)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        
        # Branch
        logits_pos = self.pointer_head(x) # [b, s, 81]
        logits_val = self.value_head(x)   # [b, s, 10]
        
        return logits_pos, logits_val
