
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

    def forward(self, x, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        b, s, d = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(b, s, self.num_heads, self.head_dim)
        k = k.view(b, s, self.num_heads, self.head_dim)
        v = v.view(b, s, self.num_heads, self.head_dim)

        # RoPE with Cache Offset
        past_len = past_kv[0].shape[2] if past_kv is not None else 0
        cos, sin = self.rope(q, seq_len=s + past_len)
        
        # Slice RoPE for current position
        cos = cos[:, past_len:past_len+s, :, :]
        sin = sin[:, past_len:past_len+s, :, :]
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose for SDPA: [b, h, s, d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Update KV Cache
        if past_kv is not None:
            k_cache, v_cache = past_kv
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
            
        current_kv = (k, v)

        # [FIX] Determine if causal masking is needed
        # If sequence length s > 1, we are processing a chunk (Training or Prompt), so we need causal mask.
        # If s == 1, we are generating token-by-token. All keys in 'k' are valid history/present.
        use_causal_mask = (s > 1)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=use_causal_mask)
        
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.o_proj(out), current_kv

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

    def forward(self, x, conv_cache: Optional[torch.Tensor] = None):
        # x: [b, s, d]
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        x_ffn = self.act(gate) * up  # [b, s, inter]
        
        # CAUSAL CONVOLUTION WITH CACHE
        x_conv_in = x_ffn.transpose(1, 2) # [b, inter, s]
        
        if conv_cache is not None:
            # Inference Mode: Prepend last input from cache
            # conv_cache: [b, inter, kernel_size-1] (Previous inputs)
            # For kernel=2, cache is just the single previous token.
            x_conv_in = torch.cat([conv_cache, x_conv_in], dim=2)
            
            # No padding needed as we have history
            out_conv = self.dwconv(x_conv_in)
            
            # Save new cache (last k-1 inputs of current sequence)
            new_conv_cache = x_conv_in[:, :, -(self.kernel_size-1):]
        else:
            # Training Mode / First Step: Use Pad
            x_pad = F.pad(x_conv_in, (self.kernel_size - 1, 0))
            out_conv = self.dwconv(x_pad)
            
            # If we want to support starting cache generation from full sequence:
            new_conv_cache = x_conv_in[:, :, -(self.kernel_size-1):] if self.training is False else None
        
        out_conv = self.act(out_conv)
        out_conv = out_conv.transpose(1, 2) # Back to [b, s, inter]
        
        return self.down_proj(out_conv), new_conv_cache

class URMBlock(nn.Module):
    def __init__(self, config: URMConfig):
        super().__init__()
        self.rms_norm_eps = 1e-6
        self.self_attn = CausalAttention(config)
        self.mlp = ConvSwiGLU_AR(config) # "Short Convolution"
        
    def forward(self, x, layer_past=None, use_cache=False):
        # layer_past: (attn_past, conv_past)
        attn_past = layer_past[0] if layer_past is not None else None
        conv_past = layer_past[1] if layer_past is not None else None
        
        # Attention
        norm_x = RMSNorm(x.shape[-1], self.rms_norm_eps)(x)
        attn_out, attn_new = self.self_attn(norm_x, past_kv=attn_past)
        x = x + attn_out
        
        # ConvSwiGLU
        norm_x = RMSNorm(x.shape[-1], self.rms_norm_eps)(x)
        mlp_out, conv_new = self.mlp(norm_x, conv_cache=conv_past)
        x = x + mlp_out
        
        current_cache = (attn_new, conv_new) if use_cache else None
        return x, current_cache

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

    def forward(self, token_ids, use_cache=False, past_caches=None):
        # token_ids: [b, s]
        # past_caches: List[List[Tuple[Tensor, Tensor]]] -> [Recurrence][Layer] -> (Attn, Conv)
        
        x = self.token_emb(token_ids)
        b, s, d = x.shape
        
        # TBPTL: Freeze gradients for the first `n_freeze` steps of recurrence
        n_freeze = 2 
        
        new_caches = [] if use_cache else None

        # Recurrence Loop
        for step in range(self.config.n_recurrence):
            # TBPTL Logic
            if step < n_freeze and self.training: # Apply only during training
                x = x.detach()

            # Loop Coordinate Embedding
            step_idx = torch.tensor([step], device=x.device)
            loop_signal = self.loop_emb(step_idx).unsqueeze(1) # [1, 1, d]
            x = x + loop_signal
            
            step_cache_out = []
            step_past_caches = past_caches[step] if past_caches is not None else None
            
            # Layer Loop
            for layer_idx, layer in enumerate(self.layers):
                layer_past = step_past_caches[layer_idx] if step_past_caches is not None else None
                
                x, layer_new = layer(x, layer_past=layer_past, use_cache=use_cache)
                
                if use_cache:
                    step_cache_out.append(layer_new)
            
            if use_cache:
                new_caches.append(step_cache_out)
                
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        if use_cache:
            return logits, new_caches
        else:
            return logits

    @torch.no_grad()
    def generate_fast(self, initial_board: str, max_actions: int = 81,
                      temperature: float = 0.0, device: str = 'cpu'):
        """
        Stateful Generation (Efficient KV Cache).
        Processes only the new token at each step.
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

        # 1. First Forward (Prompt Processing)
        start_tokens = [SOS_TOKEN] + given_tokens
        tokens = torch.tensor([start_tokens], device=device) # [1, seq_len]
        
        # Run forward with use_cache=True to build initial cache
        logits, past_caches = self.forward(tokens, use_cache=True, past_caches=None)
        
        actions = []
        valid_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool, device=device)
        for cell_id in range(81):
            if cell_id not in filled:
                start_tok = cell_id * 9
                valid_mask[start_tok:start_tok + 9] = True

        # Last token input for the loop
        next_input = tokens[:, -1:] # [1, 1] - Start loop from the last token result
        
        for _ in range(max_actions):
            # We already have logits for the last token from the previous step
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

            # Forward only the NEW token
            logits, past_caches = self.forward(next_token, use_cache=True, past_caches=past_caches)

            if len(filled) == 81:
                break

        return actions, board

    # Keep original generate for verification if needed
    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.generate_fast(*args, **kwargs)
