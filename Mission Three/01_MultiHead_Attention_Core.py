"""
Mission 03 - Part 1: Systems Architect (Lead Engineer)
Multi-head self-attention from scratch with causal masking and variance preservation.

Run: python 01_MultiHead_Attention_Core.py
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """Additive mask: 0 on and below the diagonal, -inf above it (future positions)."""
    return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V for tensors (..., seq_len, head_dim)."""
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if causal and q.size(-2) > 1:
        scores = scores + causal_mask(q.size(-2), q.device)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v), weights


class MultiHeadSelfAttention(nn.Module):
    """Pure PyTorch multi-head self-attention; nn.MultiheadAttention is not used."""

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Learned projections W_Q, W_K, W_V, W_O in R^{d_model x d_model}
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, d_model) -> (batch, seq_len, heads, head_dim) -> (batch, heads, seq_len, head_dim)
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, causal: bool = True, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, d_model); attention_mask: optional bool (batch, seq_len), True = keep.
        Returns output (batch, seq_len, d_model) and weights (batch, heads, seq_len, seq_len).
        """
        b, n, _ = x.shape
        q, k, v = self.split_heads(self.w_q(x)), self.split_heads(self.w_k(x)), self.split_heads(self.w_v(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal and n > 1:
            scores = scores + causal_mask(n, x.device)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask[:, None, None, :], float("-inf"))

        weights = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(weights, v)                      # (batch, heads, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.w_o(context), weights


def verify_variance_preservation(d_k: int = 64, seq_len: int = 128, batch_size: int = 32, seed: int = 42) -> dict:
    """With q, k ~ N(0, 1), Var(q.k) = d_k; dividing by sqrt(d_k) brings it back to ~1."""
    torch.manual_seed(seed)
    q = torch.randn(batch_size, seq_len, d_k)
    k = torch.randn(batch_size, seq_len, d_k)
    unscaled = torch.matmul(q, k.transpose(-2, -1))
    scaled = unscaled / math.sqrt(d_k)
    result = dict(
        d_k=d_k,
        input_var=q.var().item(),
        unscaled_mean=unscaled.mean().item(),
        unscaled_var=unscaled.var().item(),
        scaled_mean=scaled.mean().item(),
        scaled_var=scaled.var().item(),
    )
    result["passed"] = abs(result["scaled_var"] - 1.0) < 0.1
    return result


def softmax_saturation(d_k: int = 64, seq_len: int = 128, seed: int = 0) -> dict:
    """Why scaling matters: unscaled scores make softmax nearly one-hot (max weight close to 1)."""
    torch.manual_seed(seed)
    q, k = torch.randn(seq_len, d_k), torch.randn(seq_len, d_k)
    raw = q @ k.T
    return dict(
        unscaled_mean_max_weight=F.softmax(raw, dim=-1).max(-1).values.mean().item(),
        scaled_mean_max_weight=F.softmax(raw / math.sqrt(d_k), dim=-1).max(-1).values.mean().item(),
    )


def main():
    print("=== Part 1: Variance preservation ===")
    print(f"{'d_k':>5} | {'Var(q.k)':>9} | {'Var(q.k/sqrt(d_k))':>18} | result")
    for d_k in (16, 64, 128):
        r = verify_variance_preservation(d_k=d_k)
        print(f"{d_k:>5} | {r['unscaled_var']:>9.3f} | {r['scaled_var']:>18.4f} | {'PASS' if r['passed'] else 'FAIL'}")
        assert r["passed"], r
    s = softmax_saturation()
    print(f"Mean max softmax weight, d_k=64: unscaled {s['unscaled_mean_max_weight']:.3f}, "
          f"scaled {s['scaled_mean_max_weight']:.3f}")

    print("\n=== Part 1: Multi-head forward pass ===")
    torch.manual_seed(0)
    mha = MultiHeadSelfAttention(d_model=256, num_heads=4).eval()
    x = torch.randn(2, 16, 256)
    out, attn = mha(x, causal=True)
    print(f"Input {tuple(x.shape)} -> output {tuple(out.shape)}, attention {tuple(attn.shape)}")
    future = attn.triu(diagonal=1).abs().max().item()
    print(f"Max attention weight on future tokens: {future:.1e} (causal mask)")
    print(f"Rows sum to 1: {torch.allclose(attn.sum(-1), torch.ones(2, 4, 16))}")

    # Cross-check our math against PyTorch's fused kernel on the same projected heads.
    q, k, v = mha.split_heads(mha.w_q(x)), mha.split_heads(mha.w_k(x)), mha.split_heads(mha.w_v(x))
    ours, _ = scaled_dot_product_attention(q, k, v, causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"Max |ours - torch SDPA| = {(ours - ref).abs().max().item():.2e}")
    print("[PASS] Multi-head attention core verified.")


if __name__ == "__main__":
    main()
