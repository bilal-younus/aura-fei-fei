"""
Mission 06 - Part 1: Systems Architect (Lead Engineer)
Deterministic agent state machine with a hard step budget. Standard library only.

Run: python 01_Agent_State_Machine.py
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class AgentState(Enum):
    IDLE = "IDLE"
    PERCEIVING = "PERCEIVING"
    DELIBERATING = "DELIBERATING"
    ACTING = "ACTING"
    OBSERVING = "OBSERVING"
    REFLECTING = "REFLECTING"
    HALTED = "HALTED"


# The only legal edges. REFLECTING is the single branch point; HALTED is absorbing.
TRANSITIONS = {
    AgentState.IDLE: (AgentState.PERCEIVING,),
    AgentState.PERCEIVING: (AgentState.DELIBERATING,),
    AgentState.DELIBERATING: (AgentState.ACTING,),
    AgentState.ACTING: (AgentState.OBSERVING,),
    AgentState.OBSERVING: (AgentState.REFLECTING,),
    AgentState.REFLECTING: (AgentState.DELIBERATING, AgentState.HALTED),
    AgentState.HALTED: (),
}
MAX_BUDGET = 20


@dataclass
class Agent:
    max_steps: int = MAX_BUDGET
    state: AgentState = AgentState.IDLE
    step_count: int = 0
    mission_completed: bool = False
    halt_reason: Optional[str] = None
    transitions: List[tuple] = field(default_factory=list)

    def __post_init__(self):
        if not 1 <= self.max_steps <= MAX_BUDGET:
            raise ValueError(f"max_steps must be in 1..{MAX_BUDGET}, got {self.max_steps}")

    def _move(self, new_state: AgentState) -> AgentState:
        if new_state not in TRANSITIONS[self.state] and new_state is not AgentState.HALTED:
            raise RuntimeError(f"illegal transition {self.state.name} -> {new_state.name}")
        self.transitions.append((self.step_count, self.state.name, new_state.name))
        self.state = new_state
        return new_state

    def halt(self, reason: str) -> AgentState:
        """Force HALTED from any state (budget exhausted, mission complete or emergency tripwire)."""
        if self.state is not AgentState.HALTED:
            self.halt_reason = reason
            self._move(AgentState.HALTED)
        return self.state

    def step(self) -> AgentState:
        """Execute exactly one transition of the graph; the 21st call on a 20-step budget halts."""
        if self.state is AgentState.HALTED:
            return self.state
        if self.step_count >= self.max_steps:
            return self.halt(f"step budget of {self.max_steps} exhausted")
        self.step_count += 1
        if self.state is AgentState.REFLECTING:
            nxt = AgentState.HALTED if self.mission_completed else AgentState.DELIBERATING
            if nxt is AgentState.HALTED:
                self.halt_reason = "mission completed"
            return self._move(nxt)
        return self._move(TRANSITIONS[self.state][0])


def run_until_halt(agent: Agent, calls: int) -> List[str]:
    """Call step() up to `calls` times and return the visited state names."""
    visited = [agent.state.name]
    for _ in range(calls):
        visited.append(agent.step().name)
    return visited


def main():
    print("=== Part 1: Deterministic cycle with a 20-step budget (never completes) ===")
    agent = Agent()
    visited = run_until_halt(agent, 21)
    for i in range(1, len(visited)):
        print(f"call {i:02d}: {visited[i - 1]:>12} -> {visited[i]}")
    print(f"step_count={agent.step_count}, state={agent.state.name}, reason='{agent.halt_reason}'")
    assert agent.state is AgentState.HALTED and agent.step_count == 20

    print("\n=== Part 1: Mission completes on the first reflection ===")
    agent = Agent()
    for _ in range(5):
        agent.step()
    agent.mission_completed = True
    agent.step()
    print(" -> ".join(["IDLE"] + [t[2] for t in agent.transitions]), f"| steps={agent.step_count}")
    assert agent.state is AgentState.HALTED and agent.halt_reason == "mission completed"
    print("[PASS] State machine verified.")


if __name__ == "__main__":
    main()
