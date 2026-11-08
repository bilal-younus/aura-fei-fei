"""Handout-path launcher for the canonical Mission Six implementation."""
from pathlib import Path
import runpy,sys
root=Path(__file__).resolve().parents[2]/'Mission Six'
sys.path.insert(0,str(root))
runpy.run_path(str(root/'02_Tool_Protocol_Dispatcher.py'),run_name='__main__')
