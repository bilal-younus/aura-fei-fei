"""Regenerate the synthetic jump log, run all five scripts and save their output to evidence/."""
import contextlib
import io
import json
import platform
import sys
import time
from pathlib import Path

from common import load_part
from nebula import write_jump_dataset

ROOT = Path(__file__).resolve().parent
SCRIPTS = ["01_Agent_State_Machine.py", "02_Tool_Protocol_Dispatcher.py", "03_MDP_Trajectory_Logger.py",
           "04_Infinite_Loop_Sentinel.py", "05_System2_Warp_Pilot.py"]


def main():
    start = time.perf_counter()
    (ROOT / "evidence").mkdir(exist_ok=True)
    write_jump_dataset()
    for filename in SCRIPTS:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            load_part(filename).main()
        text = stream.getvalue()
        (ROOT / "evidence" / f"{filename[:2]}_output.txt").write_text(text)
        print(f"{filename}\n{text}", flush=True)
    (ROOT / "evidence" / "execution.json").write_text(json.dumps(dict(
        python=sys.version, platform=platform.platform(), elapsed_seconds=round(time.perf_counter() - start, 2),
        jump_dataset_seed=606, jump_dataset_rows=600, evaluation_seeds="0-999", step_budget=20, gamma=0.95), indent=2) + "\n")


if __name__ == "__main__":
    main()
