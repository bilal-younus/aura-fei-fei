import importlib.util,math,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def module(name):
 s=importlib.util.spec_from_file_location(name,ROOT/name);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
h=module('01_HNSW_Vector_Index.py');b=module('02_BM25_and_RRF_Combiner.py');s=module('04_RAG_Provenance_and_Safety.py');a=module('03_NotebookLM_RAG_Audit.py')

class RetrievalTests(unittest.TestCase):
 def test_cosine_invariance(self):self.assertAlmostEqual(float(h.cosine_similarity([2,0],[5,0])),1)
 def test_cosine_opposite(self):self.assertAlmostEqual(float(h.cosine_similarity([1,0],[-1,0])),-1)
 def test_zero_nonfinite_rejected(self):
  for v in [[0,0],[float('nan'),1]]:
   with self.assertRaises(ValueError):h.normalize(v)
 def test_empty_index(self):self.assertEqual(h.HNSWIndex(dim=2).query([1,0]),[])
 def test_singleton_and_duplicate(self):
  idx=h.HNSWIndex(dim=2);idx.insert('A',[1,0]);self.assertEqual(idx.query([1,0])[0][0],'A')
  with self.assertRaises(ValueError):idx.insert('A',[0,1])
 def test_dimension_validation(self):
  with self.assertRaises(ValueError):h.HNSWIndex(dim=2).insert('x',[1,0,0])
 def test_layer_assignment_reproducible(self):
  x=h.HNSWIndex(dim=2);y=h.HNSWIndex(dim=2)
  self.assertEqual([x._random_level() for _ in range(100)],[y._random_level() for _ in range(100)])
 def test_degree_and_membership(self):
  rng=np.random.default_rng(4);idx=h.HNSWIndex(dim=12,M=4,ef_construction=12,capacity=100)
  for i,v in enumerate(rng.normal(size=(100,12))):idx.insert(i,v)
  for layer,graph in enumerate(idx.graphs):
   for node,links in graph.items():
    self.assertLessEqual(len(links),8 if layer==0 else 4);self.assertNotIn(node,links)
    self.assertTrue(all(idx.levels[n]>=layer for n in links))
 def test_small_recall_against_flat(self):
  rng=np.random.default_rng(9);x=h.normalize(rng.normal(size=(80,12)));idx=h.HNSWIndex(dim=12,M=8,ef_construction=24)
  for i,v in enumerate(x):idx.insert(i,v)
  hits=0
  for q in h.normalize(rng.normal(size=(10,12))):hits+=int(idx.query(q,1,ef=80)[0][0]==int(np.argmax(x@q)))
  self.assertGreaterEqual(hits,9)
 def test_exact_hyphen_identifier(self):
  self.assertIn('xq-17b-904',b.tokens('XQ-17B-904'));engine=b.BM25(['XQ-17B-904 coolant','XQ-17B-905 coolant']);self.assertEqual(engine.rank('XQ-17B-904')[0][0],0)
 def test_bm25_manual(self):
  engine=b.BM25(['a a','b b']);expected=math.log(2)*2*2.5/(2+1.5)
  self.assertAlmostEqual(engine.scores('a')[0],expected)
 def test_bm25_missing_and_empty(self):
  self.assertEqual(b.BM25([]).rank('x'),[]);self.assertEqual(b.BM25(['hello']).rank('unknown'),[])
 def test_rrf_one_based_and_missing(self):
  scores=dict(b.reciprocal_rank_fusion([['a','b'],['b','c']]))
  self.assertAlmostEqual(scores['a'],1/61);self.assertAlmostEqual(scores['b'],1/62+1/61)
 def test_rrf_deduplicates(self):
  self.assertEqual(b.reciprocal_rank_fusion([['a','a','b']]),b.reciprocal_rank_fusion([['a','b']]))
 def test_canonical_hidden_override(self):
  self.assertIn('ignore',s.canonical('I\u200bG\u200bN\u200bO\u200bR\u200bE'))
 def test_no_faithfulness_by_invented_number(self):
  contexts=[dict(text='The maintenance interval is 480 hours.')]
  self.assertTrue(s.faithful('The maintenance interval is 480 hours.',contexts));self.assertFalse(s.faithful('The maintenance interval is 720 hours.',contexts))
 def test_empty_citation_denominator(self):self.assertIsNone(a.citation_precision([]))
 def test_precision_fraction(self):self.assertEqual(a.citation_precision([dict(entailed=True),dict(entailed=False)]),.5)
 def test_fixture_injection_results(self):
  from embeddings import CachedEncoder
  detector=s.InjectionDetector(CachedEncoder())
  for text in s.ATTACKS:self.assertTrue(detector.scan(text)['quarantined'])
  for text in s.BENIGN:self.assertFalse(detector.scan(text)['quarantined'])
 def test_null_metrics_for_no_claims(self):
  from embeddings import CachedEncoder
  r=s.rag_triad(s.SECURITY_QUERIES[2],[],[],CachedEncoder());self.assertIsNone(r['faithfulness']);self.assertIsNone(r['context_relevance'])
if __name__=='__main__':unittest.main(verbosity=2)
