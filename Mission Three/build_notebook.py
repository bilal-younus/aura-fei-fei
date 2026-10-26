"""Build and execute ps03_solution.ipynb from Parts 1-4. Requires nbformat, nbclient, ipykernel."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

cells = [
    md("# Mission 03 - Multi-Head Attention, RoPE & GQA KV-Cache\n"
       "Squadron Fei-Fei (Group 13), Conversational AI, BTU WS 2026/27.\n\n"
       "This notebook integrates the four station deliverables. Each part lives in its own script; "
       "the notebook imports and runs them so results here and in the terminal are identical."),
    code("import importlib.util, json, platform, torch\n"
         "from pathlib import Path\n"
         "from IPython.display import Image, display\n\n"
         "def load(name, filename):\n"
         "    spec = importlib.util.spec_from_file_location(name, Path(filename))\n"
         "    module = importlib.util.module_from_spec(spec)\n"
         "    spec.loader.exec_module(module)\n"
         "    return module\n\n"
         "mha = load('mha_core', '01_MultiHead_Attention_Core.py')\n"
         "rope = load('rope_engine', '02_RoPE_Positional_Engine.py')\n"
         "gqa = load('gqa_bench', '03_GQA_and_KVCache_Benchmark.py')\n"
         "audit = load('memory_audit', '04_memory_audit_simulation.py')\n"
         "print('torch', torch.__version__, '|', platform.platform())"),

    md("## Part 1 - Systems Architect: scaled dot-product & multi-head self-attention\n"
       "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V, built from scratch with W_Q, W_K, W_V, W_O, "
       "head splitting `(b, n, h, d) -> (b, h, n, d)` and a -inf causal mask above the diagonal."),
    code("mha.main()"),
    md("Try other attention parameters: the causal mask must keep every future weight at exactly 0."),
    code("for d_model, heads, n in [(64, 2, 8), (128, 8, 32), (512, 16, 64)]:\n"
         "    layer = mha.MultiHeadSelfAttention(d_model, heads).eval()\n"
         "    out, w = layer(torch.randn(1, n, d_model))\n"
         "    print(f'd_model={d_model:<4} heads={heads:<3} n={n:<3} out={tuple(out.shape)} '\n"
         "          f'future weight max={w.triu(diagonal=1).max().item():.1f}')"),

    md("## Part 2 - Protocol Officer: Rotary Position Embeddings\n"
       "theta_i = 10000^(-2(i-1)/d); each pair (x_2i, x_2i+1) is rotated by m * theta_i using complex "
       "multiplication. Because R_m^T R_n = R_(n-m), the score <RoPE(q,m), RoPE(k,n)> depends only on n - m."),
    code("rope.main()\n"
         "display(Image('evidence/rope_vs_alibi.png'))"),

    md("## Part 3 - Data & Payload Specialist: GQA and KV-cache\n"
       "Correctness checks run live. The full N = 1024 ... 32768 benchmark takes a few minutes on CPU, "
       "so the notebook shows the recorded run from `evidence/benchmark.json` (produced by "
       "`python 03_GQA_and_KVCache_Benchmark.py`)."),
    code("torch.manual_seed(0)\n"
         "layer = gqa.GQAAttention(d_model=256, num_q_heads=8, num_kv_heads=2).eval()\n"
         "x = torch.randn(2, 24, 256)\n"
         "with torch.no_grad():\n"
         "    full = layer(x)\n"
         "    cache = layer.new_cache(2, 24)\n"
         "    parts = [layer(x[:, :16], cache)] + [layer(x[:, t:t + 1], cache) for t in range(16, 24)]\n"
         "print('cached decode vs full forward, max diff:', (torch.cat(parts, 1) - full).abs().max().item())\n"
         "print('grouped GQA vs repeat_interleave reference:', (full - gqa.reference_gqa(layer, x)).abs().max().item())"),
    code("result = json.loads(Path('evidence/benchmark.json').read_text())\n"
         "gqa.print_benchmark(result)\n"
         "print('\\nRecorded on:', result['hardware'])"),

    md("## Part 4 - Security Sentinel: FlashAttention IO and KV-cache DoS\n"
       "Full derivation and mitigation analysis: `04_Memory_Hierarchy_and_Safety_Audit.md`."),
    code("audit.main()"),

    md("## Unified test suite"),
    code("import subprocess, sys\n"
         "r = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_mission03.py'], capture_output=True, text=True)\n"
         "print(r.stderr)"),

    md("## Squadron sign-off\n"
       "Each officer ticks their own box after reviewing their part.\n\n"
       "- [ ] Systems Architect: Bilal Younus\n"
       "- [ ] Communications & Protocol Officer: Danish Hafeez\n"
       "- [ ] Data & Payload Specialist: Anas Arshad\n"
       "- [ ] Security & Ethics Sentinel: Anas Arshad"),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
nbformat.write(nb, ROOT / "ps03_solution.ipynb")
print("Built and executed ps03_solution.ipynb")
