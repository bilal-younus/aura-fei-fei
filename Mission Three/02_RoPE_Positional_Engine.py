"""
Mission 03 - Part 2: Communications & Protocol Officer
Rotary Position Embeddings (RoPE), relative invariance check and RoPE vs ALiBi decay plot.

Run: python 02_RoPE_Positional_Engine.py   (writes evidence/rope_vs_alibi.png)
"""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent


def rope_frequencies(dim: int, base: float = 10000.0) -> torch.Tensor:
    """theta_i = base^(-2(i-1)/d) for i = 1 .. d/2: one rotation speed per 2D pair."""
    assert dim % 2 == 0, "RoPE pairs dimensions, so head_dim must be even"
    return base ** (-torch.arange(0, dim, 2).float() / dim)


def precompute_freqs_cis(dim: int, end: int, base: float = 10000.0) -> torch.Tensor:
    """cis(m * theta_i) = exp(i * m * theta_i) for positions m = 0 .. end-1; shape (end, dim // 2), complex64."""
    # Angles m * theta_i reach thousands of radians at long context; float32 would round them by ~1e-3 rad,
    # so they are formed in float64 and only the final unit complex numbers are stored as complex64.
    angles = torch.outer(torch.arange(end, dtype=torch.float64), rope_frequencies(dim, base).double())
    return torch.polar(torch.ones_like(angles), angles).to(torch.complex64)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Rotate each (x_{2i}, x_{2i+1}) pair by angle m * theta_i via complex multiplication.
    x: (..., seq_len, head_dim); freqs_cis: (seq_len, head_dim // 2) for the positions of x.
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return rotated.type_as(x)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate queries and keys with the same position table (LLaMA-style helper)."""
    return apply_rope(xq, freqs_cis), apply_rope(xk, freqs_cis)


def rope_score(q: torch.Tensor, k: torch.Tensor, m: int, n: int, freqs_cis: torch.Tensor) -> float:
    """<RoPE(q, m), RoPE(k, n)> for single vectors q, k of shape (head_dim,)."""
    return torch.dot(apply_rope(q, freqs_cis[m]), apply_rope(k, freqs_cis[n])).item()


def verify_relative_invariance(dim: int = 64, trials: int = 200, max_pos: int = 4096, seed: int = 42) -> dict:
    """Check <RoPE(q, m), RoPE(k, n)> == <RoPE(q, m + delta), RoPE(k, n + delta)> on random q, k, m, n, delta."""
    g = torch.Generator().manual_seed(seed)
    freqs_cis = precompute_freqs_cis(dim, 2 * max_pos)
    worst = 0.0
    example = None
    for _ in range(trials):
        q, k = torch.randn(dim, generator=g), torch.randn(dim, generator=g)
        m, n, delta = torch.randint(0, max_pos, (3,), generator=g).tolist()
        a, b = rope_score(q, k, m, n, freqs_cis), rope_score(q, k, m + delta, n + delta, freqs_cis)
        worst = max(worst, abs(a - b) / max(1.0, abs(a)))
        if example is None:
            example = dict(m=m, n=n, delta=delta, original=a, shifted=b)
    return dict(trials=trials, max_relative_error=worst, example=example, passed=worst < 1e-4)


def rope_self_similarity(distances, dim: int = 128, base: float = 10000.0) -> torch.Tensor:
    """
    <RoPE(x, m), RoPE(x, m + d)> / |x|^2 averaged over random x = mean_i cos(d * theta_i).
    It depends only on the distance d and shows RoPE's long-range decay (Su et al., 2021).
    """
    d = torch.as_tensor(distances, dtype=torch.float32)
    return torch.cos(torch.outer(d, rope_frequencies(dim, base))).mean(-1)


def alibi_slopes(num_heads: int) -> torch.Tensor:
    """ALiBi head slopes 2^(-8h/H) for h = 1 .. H (Press et al., 2022)."""
    return 2.0 ** (-8.0 * torch.arange(1, num_heads + 1).float() / num_heads)


def causal_weights(scores: torch.Tensor) -> torch.Tensor:
    n = scores.size(-1)
    return F.softmax(scores + torch.triu(torch.full((n, n), float("-inf")), diagonal=1), dim=-1)


def plot_rope_vs_alibi(path: Path, dim: int = 128, n: int = 64) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distances = torch.arange(0, 2049)
    rope = rope_self_similarity(distances, dim)
    shown = distances[1:]
    slopes = alibi_slopes(8)

    # Heatmaps use identical content at every position, so only position shapes the weights.
    torch.manual_seed(0)
    x = torch.randn(dim)
    x = x / x.norm() * dim ** 0.5                     # |x|^2 / sqrt(d) = sqrt(d) logit at distance 0
    rotated = apply_rope(x.expand(n, dim), precompute_freqs_cis(dim, n))
    rope_w = causal_weights(rotated @ rotated.T / dim ** 0.5)
    pos = torch.arange(n).float()
    alibi_w = causal_weights(-slopes[3] * (pos[:, None] - pos[None, :]).clamp(min=0))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(shown, rope[1:], label=f"RoPE: mean cos(d*theta_i), head_dim={dim}", color="#1f4e79", lw=1.2)
    for h, color in ((0, "#d62728"), (3, "#ff7f0e"), (7, "#2ca02c")):
        ax.plot(shown, torch.exp(-slopes[h] * shown), "--", color=color,
                label=f"ALiBi exp(-m*d), m=1/{1 / slopes[h]:.0f}")
    ax.set_xscale("log")
    ax.set_xlabel("relative distance d (tokens, log scale)")
    ax.set_ylabel("relative score factor (1 at d=0)")
    ax.set_title("Distance decay: RoPE vs ALiBi")
    ax.axhline(0, color="grey", lw=0.5)
    ax.legend(fontsize=7, loc="lower left")
    for ax, w, title in ((axes[1], rope_w, "RoPE causal weights (same token everywhere)"),
                         (axes[2], alibi_w, f"ALiBi causal weights (m=1/{1 / slopes[3]:.0f})")):
        im = ax.imshow(w, cmap="viridis")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {int(d): round(rope[d].item(), 4) for d in (0, 1, 4, 16, 64, 256, 1024, 2048)}


def main():
    print("=== Part 2: RoPE relative shift invariance ===")
    r = verify_relative_invariance()
    e = r["example"]
    print(f"Example: <RoPE(q,{e['m']}),RoPE(k,{e['n']})> = {e['original']:.6f}; "
          f"shifted by delta={e['delta']}: {e['shifted']:.6f}")
    print(f"{r['trials']} random (q, k, m, n, delta) trials, max relative error {r['max_relative_error']:.2e}")
    assert r["passed"], r

    x = torch.randn(2, 4, 32, 64)
    norms_kept = torch.allclose(apply_rope(x, precompute_freqs_cis(64, 32)).norm(dim=-1), x.norm(dim=-1), atol=1e-4)
    print(f"Rotation preserves vector norms: {norms_kept}")

    print("\n=== Part 2: Distance decay (RoPE self-similarity, head_dim=128) ===")
    decay = plot_rope_vs_alibi(ROOT / "evidence" / "rope_vs_alibi.png")
    print(" | ".join(f"d={d}: {v:+.3f}" for d, v in decay.items()))
    print(f"ALiBi slopes for 8 heads: {[f'1/{1 / s:.0f}' for s in alibi_slopes(8).tolist()]}")
    print("Plot saved: evidence/rope_vs_alibi.png")
    print("[PASS] RoPE engine verified.")


if __name__ == "__main__":
    main()
