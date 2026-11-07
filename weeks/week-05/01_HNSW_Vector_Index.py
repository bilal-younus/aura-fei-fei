"""Handout-path launcher for the canonical Mission Five implementation."""
from pathlib import Path
import runpy,sys
root=Path(__file__).resolve().parents[2]/'Mission Five'
sys.path.insert(0,str(root))
runpy.run_path(str(root/'01_HNSW_Vector_Index.py'),run_name='__main__')
