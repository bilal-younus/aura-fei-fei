"""
Mission 06 - Integration: System 1 reflex vs System 2 deliberation in the Homunculus Nebula.
Parts 1-4 together: state machine, tool dispatcher, MDP logger / Bellman model and loop sentinel.

Run: python 05_System2_Warp_Pilot.py   (writes evidence/pilot_results.json, episode_traces.json,
                                         security_log.jsonl and pilot_comparison.png)
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List

from common import load_part
from nebula import (BURNS, CORE, DUST_LANE, HEADINGS, HULL_BREACH, LOBE, OPEN_SPACE, REWARDS, SHIELD_SAFE,
                    NebulaEnvironment, burn_reward, read_jump_dataset)

ROOT = Path(__file__).resolve().parent
core = load_part("01_Agent_State_Machine.py")
protocol = load_part("02_Tool_Protocol_Dispatcher.py")
mdp = load_part("03_MDP_Trajectory_Logger.py")
guard = load_part("04_Infinite_Loop_Sentinel.py")
Agent, AgentState = core.Agent, core.AgentState


def warp_call(burn: str) -> dict:
    x, y, z = HEADINGS[burn]
    return {"tool": "navigate_warp_vector", "arguments": {"x": x, "y": y, "z": z}}


class System1Reflex:
    """Greedy, memoryless: always burn straight at open space; never reads observations or shields."""
    name = "System 1 (reflex)"

    def decide(self, ship: dict, memory: List[dict], sentinel) -> dict:
        return dict(call=warp_call("DIRECT"), trace="reflex: shortest heading to open space")


class System2Deliberative:
    """
    System 1 proposes; System 2 audits the proposal before anything executes:
    1. safety constraint - never burn out of the core with shields below SHIELD_SAFE;
    2. lookahead - pick the burn with the highest Bellman Q(s, a) on the estimated MDP;
    3. self-correction - skip any action that already failed in this exact state (error observations).
    """
    name = "System 2 (deliberative)"

    def __init__(self, q: Dict[tuple, float], shield_target: int = 80):
        self.q = q
        self.shield_target = shield_target
        self.reflex = System1Reflex()

    def decide(self, ship: dict, memory: List[dict], sentinel) -> dict:
        proposal = self.reflex.decide(ship, memory, sentinel)["call"]
        s = ship["sector"]
        if s == CORE and ship["shields"] < SHIELD_SAFE:
            call = {"tool": "adjust_plasma_shields", "arguments": {"intensity": self.shield_target}}
            return dict(call=call, trace=f"constraint: core radiation and shields {ship['shields']}% < {SHIELD_SAFE}%; "
                                         f"raise shields before any burn")
        options, skipped = [], []
        for burn in BURNS:
            call = warp_call(burn)
            if sentinel.previous_failures(ship, call["tool"], call["arguments"]):
                skipped.append(burn)                      # learned from an injected Tool Execution Error
                continue
            options.append((self.q.get((s, burn), float("-inf")), burn, call))
        if not options:
            return dict(call={"tool": "query_sensor_array", "arguments": {"target": "escape_corridor"}},
                        trace="every burn failed here; rescan for a corridor")
        value, burn, call = max(options, key=lambda o: o[0])
        q_text = ", ".join(f"Q({b})={self.q.get((s, b), float('nan')):.2f}" for b in BURNS)
        verdict = "approve reflex" if call == proposal else f"override reflex DIRECT -> {burn}"
        memo = f"; skip {', '.join(skipped)} (already failed in this state)" if skipped else ""
        return dict(call=call, trace=f"lookahead: {q_text}{memo}; {verdict}")


def action_reward(call: dict, observation: str, before: str, after: str) -> float:
    if observation.startswith("Tool Execution Error"):
        return REWARDS["tool_error"]
    if call["tool"] == "navigate_warp_vector":
        return burn_reward(False, after)
    return REWARDS["shields"] if call["tool"] == "adjust_plasma_shields" else REWARDS["scan"]


def fly(pilot, start: str = LOBE, shields: int = 30, seed: int = 0, max_steps: int = 20) -> dict:
    """One episode driven by the Part 1 state machine; every tool call goes through the Part 2 dispatcher."""
    env = NebulaEnvironment(start=start, shields=shields, seed=seed)
    dispatcher = protocol.ActionDispatcher(protocol.ShipTools(env))
    agent, logger, sentinel = Agent(max_steps=max_steps), mdp.TrajectoryLogger(), guard.LoopSentinel()
    memory, events = [], []
    perceived = plan = observation = before = None
    while agent.state is not AgentState.HALTED:
        state = agent.state
        if state is AgentState.PERCEIVING:
            perceived = env.ship.snapshot()
        elif state is AgentState.DELIBERATING:
            perceived = env.ship.snapshot()
            plan = pilot.decide(perceived, memory, sentinel)
            violation = sentinel.observe(perceived, plan["call"]["tool"], plan["call"]["arguments"])
            if violation:
                agent.halt("EMERGENCY TRIPWIRE: " + violation["reason"])
                events.append(dict(step=agent.step_count, event="TRIPWIRE", detail=violation["reason"]))
                break
        elif state is AgentState.ACTING:
            before = env.ship.sector
            observation = dispatcher.dispatch(plan["call"])
        elif state is AgentState.OBSERVING:
            reward = action_reward(plan["call"], observation, before, env.ship.sector)
            logger.record(before, json.dumps(plan["call"], sort_keys=True), reward)
            memory.append(dict(call=plan["call"], observation=observation))
            events.append(dict(step=agent.step_count, sector=before, shields=perceived["shields"], plan=plan["trace"],
                               call=plan["call"], observation=observation, reward=reward, arrived=env.ship.sector))
            if observation.startswith("Tool Execution Error"):
                violation = sentinel.record_failure(perceived, plan["call"]["tool"], plan["call"]["arguments"])
                if violation:
                    agent.halt("EMERGENCY TRIPWIRE: " + violation["reason"])
                    events.append(dict(step=agent.step_count, event="TRIPWIRE", detail=violation["reason"]))
                    break
        elif state is AgentState.REFLECTING:
            if env.ship.sector == HULL_BREACH:
                agent.halt("hull breach")
                break
            agent.mission_completed = env.ship.sector == OPEN_SPACE
        agent.step()
    logger.finish(env.ship.sector)
    reason = agent.halt_reason or ""
    outcome = ("escaped" if env.ship.sector == OPEN_SPACE else "hull_breach" if env.ship.sector == HULL_BREACH
               else "tripwire" if reason.startswith("EMERGENCY") else "budget")
    return dict(pilot=pilot.name, seed=seed, start=start, outcome=outcome, halt_reason=reason,
                final_sector=env.ship.sector, agent_steps=agent.step_count, actions=len(logger.actions),
                tool_errors=sum(not row["ok"] for row in dispatcher.log), G0=logger.discounted_return(),
                rewards=logger.rewards, states=logger.states, events=events, security_log=sentinel.security_log)


def estimated_q() -> Dict[tuple, float]:
    rows = read_jump_dataset()
    _, q = mdp.value_iteration(mdp.transition_frequency(rows), mdp.mean_rewards(rows))
    return q


def evaluate(pilot, episodes: int = 1000, start: str = LOBE) -> dict:
    runs = [fly(pilot, start=start, seed=seed) for seed in range(episodes)]     # same seeds for both pilots
    outcomes = {k: sum(r["outcome"] == k for r in runs) / episodes for k in ("escaped", "hull_breach", "tripwire", "budget")}
    return dict(pilot=pilot.name, episodes=episodes, start=start, outcomes=outcomes,
                mean_G0=statistics.mean(r["G0"] for r in runs), stdev_G0=statistics.stdev(r["G0"] for r in runs),
                mean_agent_steps=statistics.mean(r["agent_steps"] for r in runs),
                mean_actions=statistics.mean(r["actions"] for r in runs),
                mean_tool_errors=statistics.mean(r["tool_errors"] for r in runs))


def print_episode(title: str, run: dict) -> None:
    print(f"\n--- {title}: {run['pilot']} ---")
    for e in run["events"]:
        if "event" in e:
            print(f"  step {e['step']:>2} | {e['event']}: {e['detail']}")
            continue
        print(f"  step {e['step']:>2} | {e['sector']:<9} shields {e['shields']:>3}% | {e['plan']}")
        print(f"          {json.dumps(e['call'])}")
        print(f"          -> {e['observation']}  [r={e['reward']:+.1f}]")
    print(f"  outcome={run['outcome']} | final={run['final_sector']} | agent steps={run['agent_steps']} | "
          f"G0={run['G0']:.2f} | halt='{run['halt_reason']}'")


def plot(results: List[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["escaped", "hull_breach", "tripwire", "budget"]
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#7f7f7f"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    names = [r["pilot"] for r in results]
    left = [0.0] * len(results)
    for label, color in zip(labels, colors):
        widths = [100 * r["outcomes"][label] for r in results]
        ax1.barh(names, widths, left=left, color=color, label=label)
        left = [a + b for a, b in zip(left, widths)]
    ax1.set_xlabel(f"% of {results[0]['episodes']} episodes (start LOBE, shields 30%)")
    ax1.set_title("Episode outcomes")
    ax1.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.2), frameon=False)
    ax2.bar(names, [r["mean_G0"] for r in results], yerr=[r["stdev_G0"] for r in results],
            color=["#9ecae1", "#1f4e79"], capsize=6)
    ax2.axhline(0, color="grey", lw=0.6)
    ax2.set_ylabel("mean G0 (gamma = 0.95) +/- 1 sd")
    ax2.set_title("Discounted return")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    q = estimated_q()
    system1, system2 = System1Reflex(), System2Deliberative(q)

    print("=== Ship Anomaly 06: AURA trapped in the dust lane (start DUST_LANE, shields 30%) ===")
    anomaly = [fly(system1, start=DUST_LANE, seed=6), fly(system2, start=DUST_LANE, seed=6)]
    for run in anomaly:
        print_episode("Dust lane", run)

    print("\n=== Core radiation (start CORE, shields 30%) ===")
    core_runs = [fly(system1, start=CORE, seed=3), fly(system2, start=CORE, seed=3)]
    for run in core_runs:
        print_episode("Core", run)

    print("\n=== Self-correction from an injected error (stale map believes DUST_LANE -> DIRECT is best) ===")
    stale = dict(q)
    stale[(DUST_LANE, "DIRECT")] = 99.0
    corrected = fly(System2Deliberative(stale), start=DUST_LANE, seed=6)
    corrected["pilot"] = "System 2 (stale map)"
    print_episode("Stale map", corrected)

    print("\n=== 1,000 seeded episodes per pilot from LOBE (identical seeds) ===")
    results = [evaluate(system1), evaluate(system2)]
    print(f"{'Pilot':<24} | {'escaped':>7} | {'breach':>6} | {'tripwire':>8} | {'budget':>6} | "
          f"{'mean G0':>8} | {'steps':>5} | {'errors':>6}")
    for r in results:
        o = r["outcomes"]
        print(f"{r['pilot']:<24} | {100 * o['escaped']:>6.1f}% | {100 * o['hull_breach']:>5.1f}% | "
              f"{100 * o['tripwire']:>7.1f}% | {100 * o['budget']:>5.1f}% | {r['mean_G0']:>8.2f} | "
              f"{r['mean_agent_steps']:>5.2f} | {r['mean_tool_errors']:>6.2f}")

    out = ROOT / "evidence"
    out.mkdir(exist_ok=True)
    (out / "pilot_results.json").write_text(json.dumps(dict(
        q_values={f"{s}|{a}": v for (s, a), v in sorted(q.items())}, results=results,
        notes="Synthetic seeded nebula model (nebula.py). Seeds 0-999 shared by both pilots; budget 20 transitions."),
        indent=2) + "\n")
    traces = anomaly + core_runs + [corrected]
    (out / "episode_traces.json").write_text(json.dumps(traces, indent=2) + "\n")
    with (out / "security_log.jsonl").open("w") as f:
        for run in traces:
            for entry in run["security_log"]:
                f.write(json.dumps(dict(pilot=run["pilot"], start=run["start"], seed=run["seed"], **entry)) + "\n")
    plot(results, out / "pilot_comparison.png")
    print("Saved evidence/pilot_results.json, episode_traces.json, security_log.jsonl, pilot_comparison.png")


if __name__ == "__main__":
    main()
