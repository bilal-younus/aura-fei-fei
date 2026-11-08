"""Load the numbered part scripts once, so every caller shares the same classes (e.g. one AgentState enum)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = {
    "01_Agent_State_Machine.py": "ps06_agent_core",
    "02_Tool_Protocol_Dispatcher.py": "ps06_tool_protocol",
    "03_MDP_Trajectory_Logger.py": "ps06_mdp_logger",
    "04_Infinite_Loop_Sentinel.py": "ps06_loop_sentinel",
    "05_System2_Warp_Pilot.py": "ps06_warp_pilot",
}


def load_part(filename: str):
    name = NAMES[filename]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
