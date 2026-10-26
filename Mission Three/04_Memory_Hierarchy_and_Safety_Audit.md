# Mission 03 - Part 4: Security & Ethics Sentinel
## FlashAttention IO Analysis & KV-Cache Denial-of-Service Audit

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Waypoint:** Pleiades Open Cluster (`SEC-03-M45`, 444 ly from Sol)  
**Subsystem 03:** Distributed Multi-Head Attention Mesh & KV-Cache Transceiver  
**Officer Name / Squadron:** Anas Arshad / Squadron Fei-Fei (Group 13)

All numbers below come from `04_memory_audit_simulation.py` (output in `evidence/part4_output.txt` and `evidence/memory_audit.json`). They are calculated from formulas and a simulated allocator, not measured on a GPU. Latency and memory figures that were measured are in Part 3 (`evidence/benchmark.json`).

---

### Section 1: FlashAttention IO Complexity Derivation

#### 1.1 Hardware Memory Hierarchy
On current accelerators (NVIDIA A100 / H100):
* **SRAM (on-chip shared memory / L1):** about 192 KB per SM on A100 and 228 KB per SM on H100, with roughly 20 MB in total across all SMs. It is very fast (on the order of 10-20 TB/s aggregate) but tiny. The template's "~50 MB per SM" is closer to the H100's *L2* cache, which is shared by all SMs; the per-SM working memory that FlashAttention tiles into is a few hundred KB.
* **HBM (off-chip DRAM):** 40-80 GB on A100, 80 GB at about 3.35 TB/s on H100 SXM, 141 GB on H200. It is large, but roughly an order of magnitude slower than SRAM.

Attention at long context is *memory-bound*: the arithmetic is cheap compared with moving the N x N score matrix between HBM and the compute units.

#### 1.2 Standard Attention IO Complexity
Standard attention runs three separate kernels and stores the intermediate matrices in HBM. Counting HBM element reads and writes for one head (Q, K, V, O are N x d; S, P are N x N):

| Step | Reads | Writes |
|---|---|---|
| S = Q K^T | Q, K: 2Nd | S: N^2 |
| P = softmax(S) | S: N^2 | P: N^2 |
| O = P V | P: N^2, V: Nd | O: Nd |
| **Total** | | **4Nd + 4N^2** |

$$\text{IO}_{\text{Standard}} = \Theta(N \cdot d + N^2)$$

*Why N^2 saturates bandwidth beyond N ≈ 8192.* With d = 128 the Nd terms are tiny next to N^2 (N^2 / Nd = N / d = 64 at N = 8192). For one layer with 32 heads at N = 8192 in fp16, the four N^2 passes move 4 x 8192^2 x 2 B x 32 = 17.2 GB, about 5.1 ms at 3.35 TB/s. The Q/K/V/O traffic is 0.27 GB, about 0.08 ms. Over 32 layers that is about 164 ms of pure score-matrix traffic per prefill, and it grows 4x every time N doubles. Memory footprint grows the same way: one fp16 S matrix is 128 MiB at N = 8192 and **8 GiB per head at N = 65,536** (table below), so naive attention runs out of memory long before the KV-cache does.

#### 1.3 FlashAttention SRAM Tiling Derivation
FlashAttention (Dao et al., 2022) never writes S or P to HBM. Let M be the SRAM size in elements.

1. Choose block sizes B_c = ⌈M / 4d⌉ and B_r = min(B_c, d), so one block of Q, K, V, O and the B_r x B_c score tile fit in SRAM together.
2. **Outer loop** over the T_c = N / B_c blocks of K and V: load K_j, V_j into SRAM once. Total over the whole loop: **2Nd**.
3. **Inner loop** over the T_r = N / B_r blocks of Q: load Q_i, O_i and the running statistics (m_i, ℓ_i). Compute S_ij = Q_i K_j^T in SRAM, then update with the **online softmax**, which is exact rather than an approximation:
   - m_new = max(m_i, rowmax(S_ij))
   - ℓ_new = e^(m_i - m_new) ℓ_i + rowsum(e^(S_ij - m_new))
   - O_i ← [ e^(m_i - m_new) ℓ_i O_i + e^(S_ij - m_new) V_j ] / ℓ_new
   
   Then write O_i, m_i, ℓ_i back. Each outer iteration therefore streams all of Q and O once: **3Nd** per outer iteration.
4. Total: 2Nd + 3Nd · T_c = 2Nd + 3Nd · ⌈4Nd / M⌉ = **Θ(Nd + N^2 d^2 / M)**.

$$\text{IO}_{\text{Flash}} = O\left(N \cdot d + \frac{N^2 d^2}{M}\right)$$

The N^2 term is multiplied by d^2 / M. With d = 128 (d^2 = 16,384) and a 192 KiB fp16 SRAM (M = 98,304 elements), d^2 / M ≈ 0.17, so the quadratic HBM term shrinks by about 6x before constants. Dao et al. also prove a lower bound: no exact attention algorithm can use asymptotically fewer HBM accesses for all M in [d, Nd].

**Calculated element accesses per head** (d = 128, 192 KiB SRAM, fp16):

| N | Standard (M elements) | Flash (M elements) | Ratio | S matrix per head | Flash extra state (m, ℓ) |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 4.7 | 2.6 | 1.80 | 2 MiB | 8 KiB |
| 4,096 | 69.2 | 35.7 | 1.94 | 32 MiB | 32 KiB |
| 16,384 | 1,082.1 | 545.3 | 1.98 | 512 MiB | 128 KiB |
| 65,536 | 17,213.4 | 8,623.5 | 2.00 | 8,192 MiB | 512 KiB |

**Interpretation.**
- With this formula's constant of 3, the *element count* falls by about 2x. The decisive win is the memory footprint: extra memory falls from Θ(N^2) to Θ(N), the last two columns, so 64k-token attention needs 512 KiB of running statistics per head instead of an 8 GiB score matrix. It also fuses three kernels into one, so nothing round-trips through HBM.
- The mission brief's "HBM traffic from O(N^2) to O(N)" is accurate for the *extra memory*. For IO, the precise statement is Θ(Nd + N^2 d^2 / M), which is still quadratic in N but with a much smaller constant.
- FlashAttention-2 swaps the loops (outer over Q blocks), so O is written once and the work parallelizes over the sequence. FlashAttention-3 (Shah et al., 2024) targets H100: asynchronous TMA loads, warp specialization so data movement overlaps the tensor-core matmuls, and FP8 support. The IO argument above holds for all three.

---

### Section 2: KV-Cache Denial-of-Service (DoS) Simulation

#### 2.1 Threat Scenario
An unauthenticated or rogue client opens concurrent streaming sessions and drives each one to the maximum context N = 65,536, using long pasted telemetry plus "continue" requests. The target is one 80 GB accelerator serving a LLaMA-3-8B-sized model (32 layers, 32 query heads, 8 KV heads, d_head = 128, fp16).

Memory footprint of the cache:

$$\text{Bytes} = 2 \times 2 \times L \times H_{\text{KV}} \times d_{\text{head}} \times N \times B$$

(K and V) x (2 bytes fp16) x 32 layers x 8 heads x 128 x N x B gives **128 KiB per token**, or **8 GiB per 65,536-token session**.

Memory available for the cache: 80 GiB x 0.90 (vLLM's default `gpu_memory_utilization`) minus 15.0 GiB of weights = **57.0 GiB**.

| KV layout | KiB / token | GiB per 64k session | 64k sessions before OOM |
|---|---:|---:|---:|
| MHA 32:32 | 512.0 | 32.00 | 1 |
| GQA 32:8 | 128.0 | 8.00 | 7 |
| MQA 32:1 | 16.0 | 1.00 | 57 |
| MLA (d_c = 512 + 64 RoPE) | 36.0 | 2.25 | 25 |

#### 2.2 Simulation Results
The simulated allocator receives 200 arrivals in a fixed order: 160 legitimate cadet chats of 512-4,096 tokens (seeded uniform), plus one rogue 65,536-token session after every fourth chat (40 in total). Sessions are admitted in order until an allocation fails.

| Policy | KV layout | First OOM at arrival | Cadets admitted | Rogue admitted | Reserved | Wasted |
|---|---|---:|---:|---:|---:|---:|
| Contiguous max-length buffers | GQA 32:8 | **8** | 6 / 160 | 1 / 40 | 56.0 GiB | **82.40%** |
| PagedAttention (16-token blocks) | GQA 32:8 | 35 | 28 / 160 | 6 / 40 | 57.0 GiB | 0.05% |
| Paged + per-client quota (2 x 64k) | GQA 32:8 | 167 | 133 / 160 | 2 / 40 | 57.0 GiB | 0.22% |
| PagedAttention | MLA | 111 | 89 / 160 | 22 / 40 | 57.0 GiB | 0.04% |
| Paged + per-client quota | MLA | none | 160 / 160 | 2 / 40 | 18.3 GiB | 0.23% |

**Point of OOM panic.** A server that reserves a contiguous 65,536-token buffer per session fills its 57 GiB budget with **7 sessions**. The 8th request is the first failed allocation. In a naive PyTorch server without admission control, that failure is a `torch.cuda.OutOfMemoryError` inside the decoding loop, which typically kills the worker process and *every* session on it, including legitimate astrogation traffic. Two things make it worse:
- 82% of the reserved memory is empty: short chats hold 8 GiB each, so the attacker barely has to try.
- Without FlashAttention, a single 64k-token *prefill* already needs 8 GiB per head for the S matrix, so one request can trigger OOM on its own (Section 1).

---

### Section 3: Modern Architectural Mitigations

#### 3.1 PagedAttention (vLLM)
PagedAttention (Kwon et al., 2023) stores each sequence's KV-cache in fixed-size blocks (16 tokens here) that need not be contiguous. A per-sequence *block table* maps logical token blocks to physical blocks, exactly like an operating system's page table.

* **Internal fragmentation** is limited to the unfilled part of each sequence's last block (at most 15 tokens). Blocks are allocated on demand instead of reserving max-length buffers, so waste falls from 82.40% to 0.05% in our simulation. Kwon et al. report 60-80% KV waste in earlier systems and under 4% with vLLM.
* **External fragmentation** disappears because every block is the same size: any free block can serve any sequence.
* **Sharing:** copy-on-write blocks let parallel samples and common system prompts share one physical copy of the prefix.
* **Limit:** paging removes *waste*, not *demand*. Rogue sessions that genuinely fill 64k tokens still exhaust memory: with paging alone, OOM comes at arrival 35 and only 28/160 cadets are served. Paging must be combined with admission control:
  - per-client token quotas (our quota row: 133/160 cadets, rogue capped at 2 sessions);
  - a `max_model_len` cap;
  - authentication and rate limiting;
  - preemption (vLLM swaps or recomputes preempted sequences instead of crashing).
  
  The remaining 27 refused cadets in the quota row are a true capacity limit (467k tokens of budget), not the attack.

#### 3.2 Multi-Head Latent Attention (MLA / DeepSeek-V2, V3)
MLA compresses each token's keys and values into a single low-rank latent vector, c_t^KV = W^DKV h_t ∈ R^(d_c) with d_c = 512, and caches only that latent. At attention time the per-head keys and values are reconstructed with up-projections W^UK and W^UV. At inference these projections can be absorbed into W_Q and W_O, so the full K/V are never materialized.

RoPE does not commute with that absorption, so MLA adds a small decoupled RoPE key k_t^R (64 dims, shared across heads) that is cached alongside the latent. **Cache per token per layer = d_c + d_R = 576 elements.**

*Comparison table:* KV-cache size per token, using our 32-layer fp16 model (MLA dimensions applied to it for comparison):

| Mechanism | Elements per token per layer | KiB per token (32 layers) | vs MHA |
|---|---:|---:|---:|
| MHA (32 KV heads) | 2 x 32 x 128 = 8,192 | 512.0 | 1x |
| GQA 32:8 | 2 x 8 x 128 = 2,048 | 128.0 | 4x smaller |
| MQA 32:1 | 2 x 1 x 128 = 256 | 16.0 | 32x smaller |
| MLA (d_c = 512, d_R = 64) | 576 | 36.0 | 14.2x smaller |

In DeepSeek-V3 itself (61 layers, 128 heads of 128 dims), MLA caches 576 x 61 x 2 B ≈ 68.6 KiB per token. An equivalent MHA cache would be 2 x 128 x 128 = 32,768 elements per layer, about 57x larger.

**Defensive value.** At the same memory budget MLA holds 3.6x more tokens than GQA 32:8. In the simulation, OOM moves from arrival 35 to 111, and with quotas all 160 cadets fit. Unlike MQA, MLA keeps per-head keys and values (they are reconstructed from the latent), so it saves memory with less quality loss than collapsing to one KV head.

**Costs.** MLA must be trained into the model and cannot be switched on at serving time. It adds up-projection compute, and it changes only the *size* of each token's cache, not the fact that an unbounded client can keep adding tokens. It therefore raises the attack's cost but does not replace quotas.

#### 3.3 Recommended defense in depth for the *Noosphere* mainframes
1. FlashAttention-style fused kernels, so no request ever materializes an N x N matrix.
2. PagedAttention block allocation, so memory is spent only on real tokens.
3. Per-client token quotas, a `max_model_len` cap and rate limiting at the gateway, with authentication before any long-context session.
4. Preemption and graceful refusal (an HTTP 429-style "capacity exceeded" response) instead of process-wide OOM crashes.
5. GQA or MLA architectures to cut the per-token cost of every session.

**Ethics.** Availability is a safety property: an OOM crash takes down every user's session, not only the attacker's. Quotas should be transparent and applied equally, and refusals should be logged and explained rather than silently dropping legitimate users.

**Sources:**
- Dao et al., *FlashAttention* (2022)
- Dao, *FlashAttention-2* (2023)
- Shah et al., *FlashAttention-3* (2024)
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (SOSP 2023)
- DeepSeek-AI, *DeepSeek-V2* (2024) and *DeepSeek-V3 Technical Report* (2024)
- Ainslie et al., *GQA* (2023)
- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (2019)

---

### Squadron Sign-Off
- [ ] Systems Architect: Bilal Younus
- [ ] Communications Officer: Danish Hafeez
- [ ] Data Specialist: Anas Arshad
- [ ] Security Sentinel: Anas Arshad
