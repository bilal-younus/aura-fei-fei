"""Reproduce local experiments; summarize the preserved external NotebookLM observations."""
import contextlib,importlib.util,io,json,platform,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def load(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/name);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
 start=time.perf_counter()
 for filename,entry in [('01_HNSW_Vector_Index.py','benchmark'),('02_BM25_and_RRF_Combiner.py','run'),('03_NotebookLM_RAG_Audit.py','run'),('04_RAG_Provenance_and_Safety.py','run'),('graph_assist.py','run')]:
  stream=io.StringIO()
  with contextlib.redirect_stdout(stream):getattr(load(filename),entry)()
  text=stream.getvalue();(ROOT/'evidence'/(filename[:2]+'_output.txt' if filename[:2].isdigit() else 'graph_output.txt')).write_text(text)
  print(filename+'\n'+text,flush=True)
 (ROOT/'evidence/execution.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),elapsed_seconds=time.perf_counter()-start,seed=505,benchmark_n=25000,benchmark_dimensions=768),indent=2)+'\n')
if __name__=='__main__':main()
