"""
Mission 03 - Part 3: Data & Payload Specialist
Grouped-Query Attention (GQA), stateful KV-cache and MHA / GQA / MQA benchmark.

Run: python 03_GQA_and_KVCache_Benchmark.py           (full benchmark, writes evidence/benchmark.json)
     python 03_GQA_and_KVCache_Benchmark.py --quick   (short lengths only)
"""

import argparse
import importlib.util
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("rope_engine", ROOT / "02_RoPE_Positional_Engine.py")
rope_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rope_engine)


class KVCache:
    """Pre-allocated K/V buffer; each decoding step appends the new token's (K_t, V_t) slice."""

    def __init__(self, batch_size: int, max_seq_len: int, n_kv_heads: int, head_dim: int,
                 dtype=torch.float32, device="cpu"):
        shape = (batch_size, n_kv_heads, max_seq_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_seq_len = max_seq_len
        self.seq_len = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """k_new, v_new: (batch, n_kv_heads, t, head_dim). Returns all cached K, V so far."""
        t = k_new.size(2)
        if self.seq_len + t > self.max_seq_len:
            raise RuntimeError(f"KV-cache full: {self.seq_len} + {t} > {self.max_seq_len}")
        self.k[:, :, self.seq_len:self.seq_len + t] = k_new
        self.v[:, :, self.seq_len:self.seq_len + t] = v_new
        self.seq_len += t
        return self.k[:, :, :self.seq_len], self.v[:, :, :self.seq_len]

    def used_mb(self) -> float:
        """Bytes holding real tokens."""
        return 2 * self.k[:, :, :self.seq_len].nelement() * self.k.element_size() / 2**20

    def allocated_mb(self) -> float:
        """Bytes reserved for max_seq_len (what a contiguous allocator actually holds)."""
        return 2 * self.k.nelement() * self.k.element_size() / 2**20


def repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    """(b, H_kv, n, d) -> (b, H_kv * groups, n, d): copies each KV head for its query group."""
    return torch.repeat_interleave(x, repeats=groups, dim=1)


class GQAAttention(nn.Module):
    """
    Grouped-Query Attention with H_Q query heads sharing H_KV key/value heads.
    H_KV = H_Q gives MHA; H_KV = 1 gives MQA. Optional RoPE and KV-cache.
    """

    def __init__(self, d_model: int = 4096, num_q_heads: int = 32, num_kv_heads: int = 8,
                 use_rope: bool = True, max_seq_len: int = 65536):
        super().__init__()
        assert d_model % num_q_heads == 0 and num_q_heads % num_kv_heads == 0
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.groups = num_q_heads // num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len

        self.w_q = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(num_q_heads * self.head_dim, d_model, bias=False)
        self._freqs = None

    def freqs(self, start: int, n: int, device) -> torch.Tensor:
        if self._freqs is None or self._freqs.size(0) < start + n:
            self._freqs = rope_engine.precompute_freqs_cis(self.head_dim, max(start + n, 4096)).to(device)
        return self._freqs[start:start + n]

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        """x: (b, n, d_model). With a cache, x holds only the new tokens and positions continue from the cache."""
        b, n, _ = x.shape
        start = kv_cache.seq_len if kv_cache is not None else 0
        q = self.w_q(x).view(b, n, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(b, n, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(b, n, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            f = self.freqs(start, n, x.device)
            q, k = rope_engine.apply_rope(q, f), rope_engine.apply_rope(k, f)
        if kv_cache is not None:
            k, v = kv_cache.append(k, v)
        total = k.size(2)

        # Instead of repeat_interleave, fold each query group into the row dimension:
        # (b, H_q, n, d) -> (b, H_kv, group * n, d). Every K/V head is then read once, never copied per group.
        q = q.reshape(b, self.num_kv_heads, self.groups * n, self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (b, H_kv, group * n, total)
        if n > 1:  # query i (absolute position start + i) may see keys 0 .. start + i
            mask = torch.triu(torch.full((n, total), float("-inf"), device=x.device), diagonal=start + 1)
            scores = (scores.view(b, self.num_kv_heads, self.groups, n, total) + mask).view_as(scores)
        out = torch.matmul(F.softmax(scores, dim=-1), v)                          # (b, H_kv, group * n, d)
        out = out.view(b, self.num_q_heads, n, self.head_dim).transpose(1, 2).reshape(b, n, -1)
        return self.w_o(out)

    def new_cache(self, batch_size: int, max_seq_len: int, dtype=torch.float32) -> KVCache:
        return KVCache(batch_size, max_seq_len, self.num_kv_heads, self.head_dim, dtype, self.w_k.weight.device)


def reference_gqa(layer: GQAAttention, x: torch.Tensor) -> torch.Tensor:
    """Same layer computed the textbook way with repeat_interleave and torch SDPA (used by tests)."""
    b, n, _ = x.shape
    q = layer.w_q(x).view(b, n, layer.num_q_heads, layer.head_dim).transpose(1, 2)
    k = layer.w_k(x).view(b, n, layer.num_kv_heads, layer.head_dim).transpose(1, 2)
    v = layer.w_v(x).view(b, n, layer.num_kv_heads, layer.head_dim).transpose(1, 2)
    if layer.use_rope:
        f = layer.freqs(0, n, x.device)
        q, k = rope_engine.apply_rope(q, f), rope_engine.apply_rope(k, f)
    out = F.scaled_dot_product_attention(q, repeat_kv(k, layer.groups), repeat_kv(v, layer.groups), is_causal=True)
    return layer.w_o(out.transpose(1, 2).reshape(b, n, -1))


# ---------------------------------------------------------------- benchmark

# LLaMA-3-8B-like reference model for the analytic KV-cache table.
MODEL = dict(layers=32, q_heads=32, head_dim=128, d_model=4096, bytes_per_value=2)
VARIANTS = (("MHA 32:32", 32), ("GQA 32:8 (4:1)", 8), ("MQA 32:1", 1))


def kv_cache_bytes(n_tokens: int, kv_heads: int, layers: int = 32, head_dim: int = 128,
                   bytes_per_value: int = 2, batch: int = 1) -> int:
    """2 (K and V) x bytes x layers x H_kv x head_dim x tokens x batch."""
    return 2 * bytes_per_value * layers * kv_heads * head_dim * n_tokens * batch


def time_decode_step(layer: GQAAttention, context: int, reps: int = 9) -> float:
    """Median ms to decode one token against a cache already holding `context` tokens."""
    cache = layer.new_cache(1, context + reps + 1)
    cache.k[:, :, :context].normal_()
    cache.v[:, :, :context].normal_()
    x = torch.randn(1, 1, layer.d_model)
    times = []
    with torch.no_grad():
        layer(x, cache)                           # warm-up
        for _ in range(reps):
            cache.seq_len = context
            t0 = time.perf_counter()
            layer(x, cache)
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def time_no_cache_step(layer: GQAAttention, context: int, reps: int = 3) -> float:
    """Median ms to produce one token by recomputing attention over the whole prefix (no cache)."""
    x = torch.randn(1, context + 1, layer.d_model)
    times = []
    with torch.no_grad():
        for _ in range(reps):
            t0 = time.perf_counter()
            layer(x)
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def run_benchmark(seq_lengths=(1024, 4096, 16384, 32768), tokens_per_second: int = 50) -> dict:
    torch.manual_seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rows = []
    layers = {name: GQAAttention(MODEL["d_model"], MODEL["q_heads"], kv, use_rope=True).eval()
              for name, kv in VARIANTS}
    for n in seq_lengths:
        for name, kv in VARIANTS:
            nbytes = kv_cache_bytes(n, kv)
            rows.append(dict(
                seq_len=n, variant=name, kv_heads=kv,
                kv_cache_mb_32_layers_fp16=nbytes / 2**20,
                bandwidth_gb_s_at_50_tok_s=nbytes * tokens_per_second / 1e9,
                measured_cache_mb_1_layer_fp32=kv_cache_bytes(n, kv, layers=1, bytes_per_value=4) / 2**20,
                measured_ms_per_token_1_layer_cpu=time_decode_step(layers[name], n),
            ))
    gqa = layers["GQA 32:8 (4:1)"]
    cache_vs_recompute = [dict(context=n, cached_ms=time_decode_step(gqa, n), recompute_ms=time_no_cache_step(gqa, n))
                          for n in (256, 1024, 2048)]
    return dict(
        model=MODEL, tokens_per_second=tokens_per_second, rows=rows, cache_vs_recompute=cache_vs_recompute,
        hardware=dict(platform=platform.platform(), processor=platform.processor(), torch=torch.__version__,
                      threads=torch.get_num_threads(), device="cpu"),
        notes=("KV-cache MB and GB/s are analytic for a 32-layer fp16 LLaMA-3-8B-like model (cache traffic only, "
               "weights excluded). Latency is measured: one attention layer, fp32, batch 1, CPU, median of 9."),
    )


def print_benchmark(result: dict) -> None:
    print(f"{'N':>6} | {'Variant':<15} | {'KV-cache MB':>11} | {'GB/s @50 tok/s':>14} | "
          f"{'1-layer MB':>10} | {'ms/token':>8}")
    print("-" * 82)
    for r in result["rows"]:
        print(f"{r['seq_len']:>6} | {r['variant']:<15} | {r['kv_cache_mb_32_layers_fp16']:>11.1f} | "
              f"{r['bandwidth_gb_s_at_50_tok_s']:>14.2f} | {r['measured_cache_mb_1_layer_fp32']:>10.1f} | "
              f"{r['measured_ms_per_token_1_layer_cpu']:>8.2f}")
    print("-" * 82)
    print("KV-cache MB, GB/s: 32 layers, fp16 (analytic). 1-layer MB, ms/token: measured, fp32 CPU.")
    print("\nCached decode vs full recompute (GQA 32:8, one layer, ms per new token):")
    for r in result["cache_vs_recompute"]:
        print(f"  context {r['context']:>5}: cached {r['cached_ms']:7.2f} ms | recompute {r['recompute_ms']:8.2f} ms "
              f"| speed-up x{r['recompute_ms'] / r['cached_ms']:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="only N = 1024, 4096")
    args = parser.parse_args()

    print("=== Part 3: KV-cache correctness ===")
    torch.manual_seed(0)
    layer = GQAAttention(d_model=256, num_q_heads=8, num_kv_heads=2).eval()
    x = torch.randn(2, 24, 256)
    with torch.no_grad():
        full = layer(x)
        cache = layer.new_cache(2, 24)
        prefill = layer(x[:, :16], cache)
        steps = [layer(x[:, t:t + 1], cache) for t in range(16, 24)]
        cached = torch.cat([prefill, *steps], dim=1)
        ref = reference_gqa(layer, x)
    print(f"Prefill 16 + decode 8 tokens vs full forward: max diff {(cached - full).abs().max().item():.2e}")
    print(f"Grouped GQA vs repeat_interleave + torch SDPA: max diff {(full - ref).abs().max().item():.2e}")
    print(f"Cache holds {cache.seq_len} tokens, {cache.used_mb():.3f} MB")

    print("\n=== Part 3: MHA vs GQA vs MQA benchmark ===")
    result = run_benchmark((1024, 4096) if args.quick else (1024, 4096, 16384, 32768))
    print_benchmark(result)
    if not args.quick:
        (ROOT / "evidence").mkdir(exist_ok=True)
        (ROOT / "evidence" / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
        print("Saved evidence/benchmark.json")
    print("[PASS] GQA and KV-cache verified.")


if __name__ == "__main__":
    main()
