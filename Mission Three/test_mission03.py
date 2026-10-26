import importlib.util
import math
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mha = load("mha_core", "01_MultiHead_Attention_Core.py")
rope = load("rope_engine", "02_RoPE_Positional_Engine.py")
gqa = load("gqa_bench", "03_GQA_and_KVCache_Benchmark.py")
audit = load("memory_audit", "04_memory_audit_simulation.py")


class Part1AttentionTests(unittest.TestCase):
    def test_variance_preservation(self):
        for d_k in (16, 64, 128, 256):
            r = mha.verify_variance_preservation(d_k=d_k)
            self.assertAlmostEqual(r["unscaled_var"] / d_k, 1.0, delta=0.1)
            self.assertAlmostEqual(r["scaled_var"], 1.0, delta=0.1)

    def test_shapes_and_causal_mask(self):
        torch.manual_seed(0)
        layer = mha.MultiHeadSelfAttention(d_model=64, num_heads=4).eval()
        out, w = layer(torch.randn(3, 10, 64))
        self.assertEqual(out.shape, (3, 10, 64))
        self.assertEqual(w.shape, (3, 4, 10, 10))
        self.assertEqual(w.triu(diagonal=1).abs().max().item(), 0.0)
        self.assertTrue(torch.allclose(w.sum(-1), torch.ones(3, 4, 10)))

    def test_future_tokens_do_not_change_past_outputs(self):
        torch.manual_seed(1)
        layer = mha.MultiHeadSelfAttention(d_model=32, num_heads=2).eval()
        x = torch.randn(1, 8, 32)
        y = x.clone()
        y[:, 5:] = torch.randn(1, 3, 32)
        self.assertTrue(torch.allclose(layer(x)[0][:, :5], layer(y)[0][:, :5], atol=1e-6))

    def test_matches_torch_reference(self):
        torch.manual_seed(2)
        q, k, v = (torch.randn(2, 4, 12, 16) for _ in range(3))
        for causal in (True, False):
            ours, _ = mha.scaled_dot_product_attention(q, k, v, causal=causal)
            ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            self.assertTrue(torch.allclose(ours, ref, atol=1e-5))

    def test_padding_mask(self):
        layer = mha.MultiHeadSelfAttention(d_model=16, num_heads=2).eval()
        keep = torch.tensor([[True, True, False, False]])
        _, w = layer(torch.randn(1, 4, 16), causal=False, attention_mask=keep)
        self.assertEqual(w[..., 2:].abs().max().item(), 0.0)


class Part2RopeTests(unittest.TestCase):
    def test_frequencies(self):
        theta = rope.rope_frequencies(8)
        self.assertTrue(torch.allclose(theta, torch.tensor([1.0, 10000 ** -0.25, 10000 ** -0.5, 10000 ** -0.75])))

    def test_relative_invariance(self):
        r = rope.verify_relative_invariance(trials=100)
        self.assertTrue(r["passed"], r)

    def test_score_depends_only_on_offset(self):
        torch.manual_seed(3)
        f = rope.precompute_freqs_cis(32, 200)
        q, k = torch.randn(32), torch.randn(32)
        scores = [rope.rope_score(q, k, m, m + 7, f) for m in (0, 13, 50, 150)]
        for s in scores:
            self.assertAlmostEqual(s, scores[0], places=4)

    def test_norm_preserved_and_position_zero_is_identity(self):
        x = torch.randn(2, 3, 5, 16)
        out = rope.apply_rope(x, rope.precompute_freqs_cis(16, 5))
        self.assertTrue(torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5))
        self.assertTrue(torch.allclose(out[..., 0, :], x[..., 0, :]))

    def test_matches_explicit_2d_rotation(self):
        x = torch.randn(8)
        m = 5
        out = rope.apply_rope(x, rope.precompute_freqs_cis(8, m + 1)[m])
        for i, t in enumerate(rope.rope_frequencies(8)):
            c, s = math.cos(m * t), math.sin(m * t)
            a, b = x[2 * i].item(), x[2 * i + 1].item()
            self.assertAlmostEqual(out[2 * i].item(), a * c - b * s, places=5)
            self.assertAlmostEqual(out[2 * i + 1].item(), a * s + b * c, places=5)

    def test_long_range_decay(self):
        s = rope.rope_self_similarity([0, 16, 256, 2048])
        self.assertAlmostEqual(s[0].item(), 1.0, places=6)
        self.assertTrue(s[0] > s[1] > s[2] > s[3])


class Part3GqaCacheTests(unittest.TestCase):
    def test_matches_repeat_interleave_reference(self):
        torch.manual_seed(4)
        for kv in (8, 2, 1):                      # MHA, GQA 4:1, MQA
            layer = gqa.GQAAttention(d_model=128, num_q_heads=8, num_kv_heads=kv).eval()
            x = torch.randn(2, 12, 128)
            self.assertTrue(torch.allclose(layer(x), gqa.reference_gqa(layer, x), atol=1e-5))

    def test_cached_decoding_equals_full_forward(self):
        torch.manual_seed(5)
        layer = gqa.GQAAttention(d_model=128, num_q_heads=8, num_kv_heads=2).eval()
        x = torch.randn(1, 20, 128)
        cache = layer.new_cache(1, 20)
        with torch.no_grad():
            parts = [layer(x[:, :6], cache)] + [layer(x[:, t:t + 1], cache) for t in range(6, 20)]
            self.assertTrue(torch.allclose(torch.cat(parts, 1), layer(x), atol=1e-5))
        self.assertEqual(cache.seq_len, 20)

    def test_cache_memory_and_overflow(self):
        cache = gqa.KVCache(1, 8, n_kv_heads=2, head_dim=4)
        cache.append(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))
        self.assertAlmostEqual(cache.used_mb(), 2 * 2 * 3 * 4 * 4 / 2**20)
        self.assertAlmostEqual(cache.allocated_mb(), 2 * 2 * 8 * 4 * 4 / 2**20)
        with self.assertRaises(RuntimeError):
            cache.append(torch.randn(1, 2, 6, 4), torch.randn(1, 2, 6, 4))

    def test_kv_cache_sizes(self):
        mha_b = gqa.kv_cache_bytes(4096, 32)
        self.assertEqual(mha_b, 2 * 2 * 32 * 32 * 128 * 4096)
        self.assertEqual(mha_b // gqa.kv_cache_bytes(4096, 8), 4)
        self.assertEqual(mha_b // gqa.kv_cache_bytes(4096, 1), 32)


class Part4AuditTests(unittest.TestCase):
    def test_io_formulas(self):
        self.assertEqual(audit.io_standard(1024, 64), 4 * 1024 * 64 + 4 * 1024 ** 2)
        for n in (4096, 65536):
            self.assertLess(audit.io_flash(n, 128, 98304), audit.io_standard(n, 128))

    def test_kv_per_token(self):
        self.assertEqual(audit.kv_bytes_per_token("GQA 32:8"), 128 * 1024)
        self.assertEqual(audit.kv_bytes_per_token("MLA d_c=512+64"), 576 * 32 * 2)

    def test_mitigations_admit_more_legitimate_sessions(self):
        cadets = {p: audit.simulate(p)["admitted"]["cadet"] for p in ("contiguous", "paged", "paged+quota")}
        self.assertLess(cadets["contiguous"], cadets["paged"])
        self.assertLess(cadets["paged"], cadets["paged+quota"])
        self.assertLessEqual(audit.simulate("paged+quota")["admitted"]["rogue"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
