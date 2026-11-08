"""Build and execute ps06_solution.ipynb from the five scripts. Requires nbformat, nbclient, ipykernel."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

cells = [
    md("# Mission 06 - The Autonomous Warp State Machine & Deliberation Loop\n"
       "Squadron Fei-Fei (Group 13), Conversational AI, BTU WS 2026/27. Sector SEC-06-CAR (Eta Carinae).\n\n"
       "Each station's code lives in its numbered script; this notebook imports and runs them, then ends with "
       "the four `assert` test suites required by the brief. Standard library only, apart from matplotlib for one plot."),
    code("import sys, platform\n"
         "from IPython.display import Image, display\n"
         "from common import load_part\n"
         "import nebula\n"
         "core = load_part('01_Agent_State_Machine.py')\n"
         "protocol = load_part('02_Tool_Protocol_Dispatcher.py')\n"
         "mdp = load_part('03_MDP_Trajectory_Logger.py')\n"
         "guard = load_part('04_Infinite_Loop_Sentinel.py')\n"
         "pilot = load_part('05_System2_Warp_Pilot.py')\n"
         "print(sys.version.split()[0], '|', platform.platform())"),

    md("## Part 1 - Systems Architect: agent state machine\n"
       "`AgentState` enum with the seven required states, `@dataclass Agent`, and `step()` executing exactly one "
       "transition of IDLE -> PERCEIVING -> DELIBERATING -> ACTING -> OBSERVING -> REFLECTING -> (DELIBERATING | HALTED). "
       "The budget is at most 20 steps; the next call forces HALTED."),
    code("core.main()"),

    md("## Part 2 - Protocol Officer: tool protocol and action dispatcher\n"
       "Three tools, JSON-Schema contracts checked by a validator written here, and an exception boundary that turns "
       "every failure into an observation `Tool Execution Error: <msg>`."),
    code("protocol.main()"),

    md("## Part 3 - Data & Payload Specialist: MDP trajectory logger and discounted return\n"
       "tau = (s_0, a_0, r_0, ..., s_T), G_0 = sum gamma^t r_t with gamma = 0.95, and the empirical P(s'|s,a) from "
       "600 synthetic historical jumps (`data/historical_jumps.csv`). Bellman value iteration on the estimated model "
       "gives the Q-values that System 2 uses for lookahead."),
    code("mdp.main()"),

    md("## Part 4 - Security Sentinel: cycle detection and halting guardrail (Ship Anomaly 06)\n"
       "SHA-256 hashes of (environment state, planned action, arguments); a hash table gives O(1) repeat counts and "
       "Floyd's tortoise-and-hare finds cycles online. The emergency tripwire forces HALTED and logs a CRITICAL violation."),
    code("guard.main()"),

    md("## Integration - System 1 reflex vs System 2 deliberation\n"
       "Every episode is driven by the Part 1 state machine; actions go through the Part 2 dispatcher, rewards into "
       "the Part 3 logger and every plan past the Part 4 sentinel."),
    code("pilot.main()\n"
         "display(Image('evidence/pilot_comparison.png'))"),

    md("## Master test suite (brief section 3)"),
    code("# 1. State machine test: IDLE ... HALTED, and strict termination when forced to run 21 steps\n"
         "agent = core.Agent()\n"
         "assert agent.state is core.AgentState.IDLE\n"
         "core.run_until_halt(agent, 5)\n"
         "assert agent.state is core.AgentState.REFLECTING\n"
         "agent.mission_completed = True\n"
         "assert agent.step() is core.AgentState.HALTED\n"
         "forced = core.Agent(max_steps=20)\n"
         "for _ in range(21):\n"
         "    forced.step()\n"
         "assert forced.state is core.AgentState.HALTED and forced.step_count == 20\n"
         "print('State machine test passed:', forced.halt_reason)"),
    code("# 2. Protocol dispatch test: invalid JSON returns an error observation and does not crash\n"
         "d = protocol.ActionDispatcher(protocol.ShipTools(nebula.NebulaEnvironment()))\n"
         "obs = d.dispatch('{\"tool\": \"adjust_plasma_shields\", \"arguments\": {\"intensity\": 70}')\n"
         "assert isinstance(obs, str) and obs.startswith('Tool Execution Error: ')\n"
         "obs2 = d.dispatch({'tool': 'adjust_plasma_shields', 'arguments': {'intensity': -10}})\n"
         "assert obs2.startswith('Tool Execution Error: ValueError')\n"
         "print('Protocol dispatch test passed:', obs)"),
    code("# 3. MDP return test: rewards [10, 0, -5, 20], gamma = 0.95\n"
         "g0 = mdp.discounted_return([10, 0, -5, 20], gamma=0.95)\n"
         "expected = 10 + 0.95 * 0 + 0.95**2 * (-5) + 0.95**3 * 20\n"
         "assert abs(g0 - expected) < 1e-12 and abs(g0 - 22.635) < 1e-9\n"
         "print('MDP return test passed: G0 =', round(g0, 6))"),
    code("# 4. Sentinel cycle test: [s1, s2, s3, s1, s2, s3, ...] must be detected and halted before step 10\n"
         "mock = [('s1', 'a1'), ('s2', 'a2'), ('s3', 'a3')] * 5\n"
         "result = guard.guarded_replay(mock)\n"
         "assert result['agent'].state is core.AgentState.HALTED\n"
         "assert result['processed'] < 10 and result['agent'].step_count < 10\n"
         "print('Sentinel cycle test passed at element', result['processed'], '-', result['agent'].halt_reason)"),
    code("import subprocess\n"
         "r = subprocess.run([sys.executable, '-m', 'unittest', 'test_ps06.py'], capture_output=True, text=True)\n"
         "print(r.stderr.strip().splitlines()[-3], '|', r.stderr.strip().splitlines()[-1])"),

    md("## Squadron sign-off\n"
       "Each officer ticks their own box after reviewing their part.\n\n"
       "- [ ] Systems Architect: Bilal Younus\n"
       "- [ ] Communications & Protocol Officer: Danish Hafeez\n"
       "- [ ] Data & Payload Specialist: Anas Arshad\n"
       "- [ ] Security & Ethics Sentinel: Shahzaib Ahmed"),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
nbformat.write(nb, ROOT / "ps06_solution.ipynb")
print("Built and executed ps06_solution.ipynb")
