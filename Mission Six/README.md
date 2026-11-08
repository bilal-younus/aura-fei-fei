# Mission Six - Autonomous Warp State Machine & Deliberation Loop

Squadron Fei-Fei (Group 13), Conversational AI, BTU WS 2026/27.

Repository: https://github.com/bilal-younus/aura-fei-fei

## Run
Python 3.11+. The agent code and tests use only the standard library; no API key, model or agent framework is needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
python -m unittest -v test_ps06.py
```

Each part also runs on its own:

```bash
python 01_Agent_State_Machine.py
python 02_Tool_Protocol_Dispatcher.py
python 03_MDP_Trajectory_Logger.py
python 04_Infinite_Loop_Sentinel.py
python 05_System2_Warp_Pilot.py
```

`ps06_solution.ipynb` integrates all parts with executed outputs and ends with the four `assert` test suites from the brief. `python build_notebook.py` re-executes it. `mission-06-doc.md` is the report source; `python build_doc.py` converts it to `mission-06-doc.pdf`. `evidence/` contains script outputs, episode traces, the security log, pilot results, the plot, test output and `environment-lock.txt`.

## Files and the brief
The four parts follow `ps06_briefing.md` and the course portal: state machine, tool dispatcher, MDP logger with discounted return, and Floyd cycle detection. The instructor's starter files order Parts 3 and 4 differently. Their ideas are still covered:
- The starter's SHA-256 loop sentinel is part of `04_Infinite_Loop_Sentinel.py`.
- The starter's System 2 warp pilot is the integration script `05_System2_Warp_Pilot.py`.

`nebula.py` defines the shared environment. The brief mentions a "provided historical dataset of past jumps", but none is included in the course materials. `data/historical_jumps.csv` is therefore a synthetic, seeded log (600 jumps, seed 606), and the sector model and its probabilities are teaching fixtures, not course data.

The top-level `Mission Six` folder follows the repository's mission naming. `weeks/week-06/` provides launchers for the handout path.
