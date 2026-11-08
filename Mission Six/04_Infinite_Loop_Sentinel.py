"""
Mission 06 - Part 4: Security & Ethics Sentinel (Ship Anomaly 06)
SHA-256 state-action hashing, Floyd tortoise-and-hare cycle detection and an emergency tripwire.

Run: python 04_Infinite_Loop_Sentinel.py
"""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common import load_part

agent_core = load_part("01_Agent_State_Machine.py")
Agent, AgentState = agent_core.Agent, agent_core.AgentState


def state_action_hash(state: Any, action: str, arguments: Optional[dict] = None) -> str:
    """Deterministic SHA-256 of (environment state, planned action, arguments); key order does not matter."""
    payload = json.dumps({"state": state, "action": action, "arguments": arguments or {}}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def floyd_meeting(seq: Sequence) -> Optional[int]:
    """Tortoise at i, hare at 2i: the first i > 0 with seq[i] == seq[2i], or None."""
    for i in range(1, (len(seq) - 1) // 2 + 1):
        if seq[i] == seq[2 * i]:
            return i
    return None


def floyd_cycle(seq: Sequence) -> Optional[Tuple[int, int]]:
    """
    Floyd's algorithm on a recorded sequence x_0, x_1, ...: returns (mu, lambda) = (start of the cycle,
    cycle length) or None. Phase 1 finds a meeting distance nu (a multiple of lambda); phase 2 walks a
    tortoise from 0 and a hare from nu in lockstep to find mu; phase 3 measures lambda from mu.
    """
    nu = floyd_meeting(seq)
    if nu is None:
        return None
    mu = 0
    while seq[mu] != seq[mu + nu]:
        mu += 1
    lam = 1
    while seq[mu + lam] != seq[mu]:
        lam += 1
    return mu, lam


@dataclass
class LoopSentinel:
    """
    Online guard. Every planned (state, action) is hashed; a hash table gives O(1) repeat counts and
    the tortoise-and-hare comparison runs each time the hare reaches an even index.
    """
    repeat_threshold: int = 3
    hashes: List[str] = field(default_factory=list)
    seen: Counter = field(default_factory=Counter)
    failed: Counter = field(default_factory=Counter)
    security_log: List[dict] = field(default_factory=list)
    tripped: Optional[dict] = None

    def observe(self, state: Any, action: str, arguments: Optional[dict] = None, failed: bool = False) -> Optional[dict]:
        """Register one planned step; returns the violation record if the tripwire fires, else None."""
        h = state_action_hash(state, action, arguments)
        t = len(self.hashes)
        self.hashes.append(h)
        self.seen[h] += 1
        if t > 0 and t % 2 == 0 and self.hashes[t // 2] == h:        # tortoise t/2 meets hare t
            mu, lam = floyd_cycle(self.hashes)
            return self._trip(f"Floyd cycle detected: mu={mu}, lambda={lam}, tortoise={t // 2}, hare={t}",
                              state, action, arguments, h)
        return self.record_failure(state, action, arguments) if failed else None

    def record_failure(self, state: Any, action: str, arguments: Optional[dict] = None) -> Optional[dict]:
        """Count a failed execution of (state, action); the third identical failure fires the tripwire."""
        h = state_action_hash(state, action, arguments)
        self.failed[h] += 1
        if self.failed[h] >= self.repeat_threshold:
            return self._trip(f"same failed action in the same state {self.failed[h]} times", state, action, arguments, h)
        return None

    def _trip(self, reason: str, state: Any, action: str, arguments: Optional[dict], h: str) -> Optional[dict]:
        if self.tripped is not None:
            return None
        self.tripped = dict(level="CRITICAL", event="EMERGENCY_TRIPWIRE", step=len(self.hashes) - 1, reason=reason,
                            action=action, arguments=arguments or {}, state=state, hash=h[:16], repeats=self.seen[h])
        self.security_log.append(self.tripped)
        return self.tripped

    def previous_failures(self, state: Any, action: str, arguments: Optional[dict] = None) -> int:
        """O(1) lookup used by deliberation to avoid re-proposing a known failure."""
        return self.failed[state_action_hash(state, action, arguments)]


def guarded_replay(sequence: Sequence[Tuple[Any, str]], agent: Optional[Agent] = None,
                   sentinel: Optional[LoopSentinel] = None) -> Dict[str, Any]:
    """Feed a state-action sequence to an agent, one agent step per element, until the tripwire halts it."""
    agent = agent or Agent()
    sentinel = sentinel or LoopSentinel()
    processed = 0
    for state, action in sequence:
        if agent.state is AgentState.HALTED:
            break
        agent.step()
        processed += 1
        violation = sentinel.observe(state, action)
        if violation:
            agent.halt("EMERGENCY TRIPWIRE: " + violation["reason"])
    return dict(processed=processed, agent=agent, sentinel=sentinel)


def main():
    print("=== Part 4: Floyd on a periodic state-action trace [s1, s2, s3, s1, s2, s3, ...] ===")
    mock = [("s1", "scan"), ("s2", "burn"), ("s3", "shields")] * 6
    result = guarded_replay(mock)
    agent, sentinel = result["agent"], result["sentinel"]
    print(f"Elements processed: {result['processed']} of {len(mock)}")
    print(f"Agent: state={agent.state.name}, step_count={agent.step_count}, reason='{agent.halt_reason}'")
    print("Security log:", json.dumps({k: v for k, v in sentinel.tripped.items() if k != 'state'}))
    assert agent.state is AgentState.HALTED and result["processed"] < 10

    print("\n=== Part 4: Same failed action in the same state (O(1) hash counter) ===")
    sentinel = LoopSentinel()
    state = {"sector": "DUST_LANE", "shields": 30, "plasma": 100}
    args = {"x": 1.0, "y": 0.0, "z": 0.0}
    for attempt in range(1, 5):
        violation = sentinel.record_failure(state, "navigate_warp_vector", args)
        print(f"attempt {attempt}: hash {state_action_hash(state, 'navigate_warp_vector', args)[:16]}... "
              f"failures={sentinel.previous_failures(state, 'navigate_warp_vector', args)} "
              f"tripwire={'FIRED - ' + violation['reason'] if violation else 'armed'}")
        if violation:
            break

    print("\n=== Part 4: Floyd (mu, lambda) on sample sequences ===")
    for seq in ["abcabcabc", "xyabababab", "aaaa", "abcdefgh"]:
        print(f"{seq:<12} -> {floyd_cycle(seq)}")
    print("[PASS] Loop sentinel verified.")


if __name__ == "__main__":
    main()
