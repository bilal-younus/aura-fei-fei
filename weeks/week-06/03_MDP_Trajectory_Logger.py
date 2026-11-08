"""Handout-path launcher for the canonical Mission Six implementation."""
from pathlib import Path
import runpy,sys
root=Path(__file__).resolve().parents[2]/'Mission Six'
sys.path.insert(0,str(root))
runpy.run_path(str(root/'03_MDP_Trajectory_Logger.py'),run_name='__main__')
