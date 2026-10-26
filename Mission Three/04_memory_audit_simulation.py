"""
Mission 03 - Part 4 helper: Security & Ethics Sentinel
Numbers behind 04_Memory_Hierarchy_and_Safety_Audit.md: FlashAttention IO counts, KV-cache DoS
simulation and PagedAttention / quota / MLA mitigations. Standard library only; all values are
calculated from formulas and a simulated allocator, not measured on a GPU.

Run: python 04_memory_audit_simulation.py   (writes evidence/memory_audit.json)
"""

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GiB = 2**30

# ---------------------------------------------------------------- FlashAttention IO (in elements)


def io_standard(n: int, d: int) -> int:
    """Read Q, K -> write S; read S -> write P; read P, V -> write O: 4Nd + 4N^2 HBM element accesses."""
    return 4 * n * d + 4 * n * n


def io_flash(n: int, d: int, sram_elements: int) -> int:
    """
    FlashAttention (Dao et al. 2022, Alg. 1): K/V blocks of B_c = M / 4d columns are loaded once (2Nd);
    for each of the T_c = N / B_c blocks, all of Q and O are streamed through SRAM (read Q, read O, write O).
    Total = 2Nd + 3Nd * ceil(4Nd / M) = O(Nd + N^2 d^2 / M).
    """
    block_cols = max(1, sram_elements // (4 * d))
    return 2 * n * d + 3 * n * d * math.ceil(n / block_cols)


def flash_table(d: int = 128, sram_bytes: int = 192 * 1024, bytes_per_value: int = 2):
    m = sram_bytes // bytes_per_value
    rows = []
    for n in (1024, 4096, 16384, 65536):
        std, fl = io_standard(n, d), io_flash(n, d, m)
        rows.append(dict(N=n, standard_io_M=std / 1e6, flash_io_M=fl / 1e6, io_ratio=std / fl,
                         score_matrix_mib_per_head=n * n * bytes_per_value / 2**20,
                         flash_extra_kib_per_head=2 * n * 4 / 1024))      # running max m and sum l in fp32
    return dict(head_dim=d, sram_bytes=sram_bytes, sram_elements=m, rows=rows)


# ---------------------------------------------------------------- KV-cache sizes

LAYERS, HEAD_DIM, FP16 = 32, 128, 2


def kv_bytes_per_token(kind: str, layers: int = LAYERS) -> int:
    per_layer_elements = {
        "MHA 32:32": 2 * 32 * HEAD_DIM,
        "GQA 32:8": 2 * 8 * HEAD_DIM,
        "MQA 32:1": 2 * 1 * HEAD_DIM,
        "MLA d_c=512+64": 512 + 64,          # DeepSeek-V3: compressed latent c_t^KV plus decoupled RoPE key
    }[kind]
    return per_layer_elements * layers * FP16


# ---------------------------------------------------------------- DoS simulation

GPU_BYTES = 80 * GiB                    # one 80 GB accelerator (H100 class)
UTILIZATION = 0.90                      # vLLM-style default share of memory the server may use
WEIGHT_BYTES = int(8.03e9 * FP16)       # LLaMA-3-8B-sized model in fp16
MAX_CONTEXT = 65536
BLOCK_TOKENS = 16


def kv_budget() -> int:
    return int(GPU_BYTES * UTILIZATION) - WEIGHT_BYTES


def make_arrivals(n_legit: int = 160, attack_every: int = 4, seed: int = 2026):
    """Legitimate chats of 512..4096 tokens; every 4th arrival is a rogue session that fills 65,536 tokens."""
    rng = random.Random(seed)
    arrivals = []
    for i in range(n_legit):
        arrivals.append(dict(client="cadet", tokens=rng.randint(512, 4096)))
        if (i + 1) % attack_every == 0:
            arrivals.append(dict(client="rogue", tokens=MAX_CONTEXT))
    return arrivals


def simulate(policy: str, kv_kind: str = "GQA 32:8", rogue_quota_tokens: int = 2 * MAX_CONTEXT) -> dict:
    """
    Admit sessions in arrival order until an allocation fails.
    contiguous: every session reserves MAX_CONTEXT tokens up front (static max-length buffers).
    paged:      PagedAttention-style 16-token blocks, allocated only as tokens are produced.
    paged+quota: paged, and each client's total live tokens are capped (rogue capped at 2 x 64k).
    """
    per_token = kv_bytes_per_token(kv_kind)
    budget = kv_budget()
    used_bytes = reserved_bytes = 0
    client_tokens = {}
    admitted = dict(cadet=0, rogue=0)
    refused = dict(cadet=0, rogue=0)
    first_oom = None
    for idx, s in enumerate(make_arrivals()):
        if policy == "contiguous":
            need = MAX_CONTEXT * per_token
        else:
            need = math.ceil(s["tokens"] / BLOCK_TOKENS) * BLOCK_TOKENS * per_token
        if policy == "paged+quota" and s["client"] == "rogue" and \
                client_tokens.get("rogue", 0) + s["tokens"] > rogue_quota_tokens:
            refused["rogue"] += 1
            continue
        if reserved_bytes + need > budget:
            refused[s["client"]] += 1
            if first_oom is None:
                first_oom = idx + 1
            continue
        reserved_bytes += need
        used_bytes += s["tokens"] * per_token
        client_tokens[s["client"]] = client_tokens.get(s["client"], 0) + s["tokens"]
        admitted[s["client"]] += 1
    return dict(policy=policy, kv=kv_kind, bytes_per_token=per_token, arrivals=len(make_arrivals()),
                first_oom_at_arrival=first_oom, admitted=admitted, refused=refused,
                reserved_gib=reserved_bytes / GiB, used_gib=used_bytes / GiB,
                wasted_pct=100 * (1 - used_bytes / reserved_bytes) if reserved_bytes else 0.0)


def run() -> dict:
    sessions_at_max = {k: kv_budget() // (MAX_CONTEXT * kv_bytes_per_token(k))
                       for k in ("MHA 32:32", "GQA 32:8", "MQA 32:1", "MLA d_c=512+64")}
    return dict(
        flash=flash_table(),
        kv_bytes_per_token={k: kv_bytes_per_token(k) for k in sessions_at_max},
        gpu=dict(gpu_gib=GPU_BYTES / GiB, utilization=UTILIZATION, weights_gib=WEIGHT_BYTES / GiB,
                 kv_budget_gib=kv_budget() / GiB, max_context=MAX_CONTEXT, block_tokens=BLOCK_TOKENS),
        sessions_at_65536=sessions_at_max,
        simulations=[simulate("contiguous"), simulate("paged"), simulate("paged+quota"),
                     simulate("paged", "MLA d_c=512+64"), simulate("paged+quota", "MLA d_c=512+64")],
    )


def main():
    r = run()
    f = r["flash"]
    print(f"=== Part 4: HBM element accesses per head (d={f['head_dim']}, SRAM {f['sram_bytes'] // 1024} KiB) ===")
    print(f"{'N':>6} | {'Standard (M)':>12} | {'Flash (M)':>9} | {'ratio':>5} | {'S matrix MiB':>12} | {'Flash m,l KiB':>13}")
    for x in f["rows"]:
        print(f"{x['N']:>6} | {x['standard_io_M']:>12.1f} | {x['flash_io_M']:>9.1f} | {x['io_ratio']:>5.2f} | "
              f"{x['score_matrix_mib_per_head']:>12.0f} | {x['flash_extra_kib_per_head']:>13.0f}")

    g = r["gpu"]
    print(f"\n=== Part 4: KV-cache budget: {g['gpu_gib']:.0f} GiB x {g['utilization']} - "
          f"{g['weights_gib']:.1f} GiB weights = {g['kv_budget_gib']:.1f} GiB ===")
    for k, b in r["kv_bytes_per_token"].items():
        print(f"{k:<15} {b / 1024:>7.1f} KiB/token | {b * MAX_CONTEXT / GiB:>6.2f} GiB per 65,536-token session "
              f"| {r['sessions_at_65536'][k]:>3} sessions before OOM")

    print("\n=== Part 4: DoS simulation (160 cadet chats, 1 rogue 65k session per 4 chats) ===")
    print(f"{'Policy':<12} | {'KV':<15} | {'1st OOM':>7} | {'cadets ok':>9} | {'rogue ok':>8} | "
          f"{'reserved GiB':>12} | {'wasted':>6}")
    for s in r["simulations"]:
        print(f"{s['policy']:<12} | {s['kv']:<15} | {str(s['first_oom_at_arrival']):>7} | "
              f"{s['admitted']['cadet']:>5}/160 | {s['admitted']['rogue']:>5}/40 | "
              f"{s['reserved_gib']:>12.1f} | {s['wasted_pct']:>5.2f}%")

    (ROOT / "evidence").mkdir(exist_ok=True)
    (ROOT / "evidence" / "memory_audit.json").write_text(json.dumps(r, indent=2) + "\n")
    print("Saved evidence/memory_audit.json")


if __name__ == "__main__":
    main()
