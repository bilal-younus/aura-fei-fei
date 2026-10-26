# Mission 03 - Multi-head attention, RoPE and GQA KV-cache

Squadron Fei-Fei (Group 13), Conversational AI, BTU WS 2026/27.

Repository: https://github.com/bilal-younus/aura-fei-fei
Mission folder: https://github.com/bilal-younus/aura-fei-fei/tree/main/Mission%20Three
Brief: `mission-03/ps03_briefing.md` in https://github.com/Unisvet/couse-c-ai

## Run
Python 3.9+ with PyTorch. No API key or GPU is needed.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python 01_MultiHead_Attention_Core.py
python 02_RoPE_Positional_Engine.py          # writes evidence/rope_vs_alibi.png
python 03_GQA_and_KVCache_Benchmark.py       # N up to 32,768; a few minutes on CPU (--quick for N <= 4096)
python 04_memory_audit_simulation.py
python -m unittest -v test_mission03.py
```

Optional rebuilds: `python build_notebook.py` re-executes `ps03_solution.ipynb`; `python build_doc.py` turns `mission-03-doc.md` into `mission-03-doc.pdf` with headless Chrome.

## Contents
- `00_Team_Roles.md`: assigned stations.
- `01_MultiHead_Attention_Core.py`: `MultiHeadSelfAttention` from scratch, causal mask, variance preservation.
- `02_RoPE_Positional_Engine.py`: `apply_rope`, relative shift invariance, RoPE vs ALiBi decay plot.
- `03_GQA_and_KVCache_Benchmark.py`: `GQAAttention`, `KVCache`, MHA / GQA 4:1 / MQA benchmark.
- `04_Memory_Hierarchy_and_Safety_Audit.md`: FlashAttention IO derivation, KV-cache DoS audit, PagedAttention and MLA.
- `04_memory_audit_simulation.py`: calculations and simulated allocator behind the audit.
- `ps03_solution.ipynb`: executed notebook integrating all four parts.
- `test_mission03.py`: 18 unit tests.
- `mission-03-doc.md` / `mission-03-doc.pdf`: squadron report for Moodle.
- `evidence/`: terminal outputs, benchmark and audit JSON, plot, test log.

## Recorded run
26 October 2026, macOS on an Intel Core i7-9750H, Python 3.9.6, PyTorch 2.2.2, CPU only. All four scripts completed, 18/18 tests passed and the notebook executed without errors. PyTorch 2.4+ is the course standard, but 2.2.2 is the last build published for Intel Macs; the code uses only core tensor operations and also runs on newer versions (for example on Colab).

Measured latencies are one fp32 attention layer on CPU and show relative scaling only. The 32-layer KV-cache sizes, the bandwidth figures, the FlashAttention IO counts and the DoS table are calculated, not measured on a GPU.

Tools: Claude Code (Claude Opus 5.5) assisted with implementation, tests and report preparation. Station sign-offs are left for each member to tick. Submission steps are in SUBMIT.md.
