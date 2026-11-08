import math
import unittest

import nebula
from common import load_part

core = load_part("01_Agent_State_Machine.py")
protocol = load_part("02_Tool_Protocol_Dispatcher.py")
mdp = load_part("03_MDP_Trajectory_Logger.py")
guard = load_part("04_Infinite_Loop_Sentinel.py")
pilot = load_part("05_System2_Warp_Pilot.py")

Agent, AgentState = core.Agent, core.AgentState


def dispatcher(**env_kwargs):
    return protocol.ActionDispatcher(protocol.ShipTools(nebula.NebulaEnvironment(**env_kwargs)))


class Part1StateMachineTests(unittest.TestCase):
    def test_brief_idle_to_halted_and_21_steps(self):
        agent = Agent()
        self.assertEqual(core.run_until_halt(agent, 5)[-1], "REFLECTING")
        agent.mission_completed = True
        self.assertIs(agent.step(), AgentState.HALTED)
        forced = Agent(max_steps=20)
        core.run_until_halt(forced, 21)
        self.assertIs(forced.state, AgentState.HALTED)
        self.assertEqual(forced.step_count, 20)

    def test_exact_states_and_legal_edges_only(self):
        self.assertEqual([s.name for s in AgentState],
                         ["IDLE", "PERCEIVING", "DELIBERATING", "ACTING", "OBSERVING", "REFLECTING", "HALTED"])
        agent = Agent()
        core.run_until_halt(agent, 30)
        for _, before, after in agent.transitions:
            legal = core.TRANSITIONS[AgentState[before]]
            self.assertTrue(AgentState[after] in legal or after == "HALTED")

    def test_halted_is_absorbing_and_budget_validated(self):
        agent = Agent(max_steps=3)
        core.run_until_halt(agent, 10)
        self.assertEqual((agent.state, agent.step_count), (AgentState.HALTED, 3))
        for bad in (0, 21):
            with self.assertRaises(ValueError):
                Agent(max_steps=bad)


class Part2ProtocolTests(unittest.TestCase):
    def test_brief_invalid_json_returns_error_observation(self):
        d = dispatcher()
        for payload in ['{"tool": "adjust_plasma_shields", "arguments": {', "not json", "[1, 2]", "null"]:
            obs = d.dispatch(payload)
            self.assertIsInstance(obs, str)
            self.assertTrue(obs.startswith("Tool Execution Error: "), obs)

    def test_tool_value_error_is_caught(self):
        obs = dispatcher().dispatch({"tool": "adjust_plasma_shields", "arguments": {"intensity": -5}})
        self.assertEqual(obs, "Tool Execution Error: ValueError: shield intensity -5% is out of bounds; "
                              "valid range is [0, 100]")

    def test_schema_types_and_bounds(self):
        schema = protocol.TOOL_SCHEMAS["navigate_warp_vector"]
        self.assertEqual(protocol.validate({"x": 0.5, "y": 0, "z": -1}, schema), [])
        self.assertTrue(protocol.validate({"x": True, "y": 0, "z": 0}, schema))
        self.assertTrue(protocol.validate({"x": math.nan, "y": 0, "z": 0}, schema))
        self.assertTrue(protocol.validate({"x": 1.5, "y": 0, "z": 0}, schema))
        self.assertTrue(protocol.validate({"x": 0, "y": 0, "z": 0, "w": 1}, schema))
        self.assertTrue(protocol.validate({"intensity": "80"}, protocol.TOOL_SCHEMAS["adjust_plasma_shields"]))

    def test_valid_calls_change_ship_state(self):
        d = dispatcher(start=nebula.CORRIDOR, seed=0)
        self.assertTrue(d.dispatch({"tool": "adjust_plasma_shields", "arguments": {"intensity": 80}}).startswith("Observation:"))
        self.assertEqual(d.registry["adjust_plasma_shields"].__self__.env.ship.shields, 80)
        obs = d.dispatch('{"tool": "navigate_warp_vector", "arguments": {"x": -0.85, "y": 0.52, "z": 0.05}}')
        self.assertIn("CORRIDOR heading", obs)


class Part3MDPTests(unittest.TestCase):
    def test_brief_discounted_return(self):
        expected = 10 + 0.95 * 0 + 0.95 ** 2 * -5 + 0.95 ** 3 * 20
        self.assertAlmostEqual(mdp.discounted_return([10, 0, -5, 20], 0.95), expected)
        self.assertAlmostEqual(expected, 22.635)

    def test_logger_tau_and_returns_to_go(self):
        log = mdp.TrajectoryLogger()
        log.record("A", "go", 1)
        log.record("B", "go", 2)
        log.finish("C")
        self.assertEqual(log.tau(), ["A", "go", 1.0, "B", "go", 2.0, "C"])
        self.assertEqual(mdp.returns_to_go([1, 2], 0.5), [2.0, 2.0])
        with self.assertRaises(RuntimeError):
            log.record("C", "go", 0)

    def test_transition_frequency_normalized_and_close_to_truth(self):
        p_hat = mdp.transition_frequency(nebula.read_jump_dataset())
        for dist in p_hat.values():
            self.assertAlmostEqual(sum(dist.values()), 1.0)
        self.assertEqual(p_hat[("DUST_LANE", "DIRECT")], {"DUST_LANE": 1.0})
        self.assertLess(mdp.max_estimation_error(p_hat), 0.08)

    def test_manual_frequency(self):
        rows = [dict(state="s", action="a", next_state=x) for x in ("u", "u", "v", "u")]
        self.assertEqual(mdp.transition_frequency(rows), {("s", "a"): {"u": 0.75, "v": 0.25}})


class Part4SentinelTests(unittest.TestCase):
    def test_brief_cycle_halts_before_step_10(self):
        result = guard.guarded_replay([("s1", "a1"), ("s2", "a2"), ("s3", "a3")] * 5)
        self.assertIs(result["agent"].state, AgentState.HALTED)
        self.assertLess(result["processed"], 10)
        self.assertLess(result["agent"].step_count, 10)
        self.assertEqual(result["sentinel"].tripped["level"], "CRITICAL")

    def test_floyd_mu_lambda(self):
        self.assertEqual(guard.floyd_cycle("abcabcabc"), (0, 3))
        self.assertEqual(guard.floyd_cycle("xyabababab"), (2, 2))
        self.assertIsNone(guard.floyd_cycle("abcdefgh"))

    def test_hash_is_order_independent_and_state_sensitive(self):
        h = guard.state_action_hash
        self.assertEqual(h({"a": 1, "b": 2}, "go", {"x": 1, "y": 2}), h({"b": 2, "a": 1}, "go", {"y": 2, "x": 1}))
        self.assertNotEqual(h({"plasma": 100}, "go"), h({"plasma": 95}, "go"))
        self.assertEqual(len(h("s", "a")), 64)

    def test_third_identical_failure_trips(self):
        sentinel = guard.LoopSentinel()
        results = [sentinel.record_failure("DUST_LANE", "burn", {"x": 1}) for _ in range(3)]
        self.assertEqual(results[:2], [None, None])
        self.assertIn("3 times", results[2]["reason"])

    def test_progress_does_not_trip(self):
        sentinel = guard.LoopSentinel()
        self.assertTrue(all(sentinel.observe({"plasma": 100 - 5 * t}, "burn") is None for t in range(20)))


class IntegrationTests(unittest.TestCase):
    def test_system1_trapped_system2_escapes_dust_lane(self):
        q = pilot.estimated_q()
        reflex = pilot.fly(pilot.System1Reflex(), start=nebula.DUST_LANE, seed=6)
        deliberate = pilot.fly(pilot.System2Deliberative(q), start=nebula.DUST_LANE, seed=6)
        self.assertEqual(reflex["outcome"], "tripwire")
        self.assertEqual(deliberate["outcome"], "escaped")
        self.assertLessEqual(deliberate["agent_steps"], 20)

    def test_system2_raises_shields_in_core(self):
        run = pilot.fly(pilot.System2Deliberative(pilot.estimated_q()), start=nebula.CORE, seed=3)
        self.assertEqual(run["events"][0]["call"]["tool"], "adjust_plasma_shields")

    def test_error_observation_drives_self_correction(self):
        q = dict(pilot.estimated_q())
        q[(nebula.DUST_LANE, "DIRECT")] = 99.0
        run = pilot.fly(pilot.System2Deliberative(q), start=nebula.DUST_LANE, seed=6)
        self.assertTrue(run["events"][0]["observation"].startswith("Tool Execution Error"))
        self.assertEqual(run["events"][1]["call"], pilot.warp_call("CORRIDOR"))
        self.assertEqual(run["outcome"], "escaped")


if __name__ == "__main__":
    unittest.main(verbosity=2)
