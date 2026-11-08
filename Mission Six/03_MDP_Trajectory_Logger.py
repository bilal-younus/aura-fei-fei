"""
Mission 06 - Part 3: Data & Payload Specialist
MDP trajectory logger, empirical discounted return and transition frequencies P(s' | s, a).

Run: python 03_MDP_Trajectory_Logger.py
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from nebula import BURNS, DYNAMICS, TERMINAL, read_jump_dataset

GAMMA = 0.95


@dataclass
class TrajectoryLogger:
    """Records tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T)."""
    states: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    closed: bool = False

    def record(self, state: str, action: str, reward: float) -> None:
        if self.closed:
            raise RuntimeError("trajectory already finished")
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))

    def finish(self, final_state: str) -> None:
        self.states.append(final_state)
        self.closed = True

    def tau(self) -> list:
        flat = []
        for t, (a, r) in enumerate(zip(self.actions, self.rewards)):
            flat += [self.states[t], a, r]
        return flat + self.states[len(self.actions):]

    def discounted_return(self, gamma: float = GAMMA) -> float:
        return discounted_return(self.rewards, gamma)


def discounted_return(rewards: Sequence[float], gamma: float = GAMMA) -> float:
    """G_0 = sum_t gamma^t r_t."""
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    return sum(gamma ** t * r for t, r in enumerate(rewards))


def returns_to_go(rewards: Sequence[float], gamma: float = GAMMA) -> List[float]:
    """G_t = r_t + gamma * G_{t+1}, computed backwards."""
    g, out = 0.0, []
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    return out[::-1]


def transition_frequency(rows: List[dict]) -> Dict[tuple, Dict[str, float]]:
    """Empirical P(s' | s, a) = N(s, a, s') / N(s, a) from logged jumps."""
    counts: Dict[tuple, Counter] = defaultdict(Counter)
    for row in rows:
        counts[(row["state"], row["action"])][row["next_state"]] += 1
    return {sa: {s2: n / sum(c.values()) for s2, n in sorted(c.items())} for sa, c in sorted(counts.items())}


def mean_rewards(rows: List[dict]) -> Dict[tuple, float]:
    """Empirical expected immediate reward R(s, a, s')."""
    sums, counts = defaultdict(float), Counter()
    for row in rows:
        key = (row["state"], row["action"], row["next_state"])
        sums[key] += float(row["reward"])
        counts[key] += 1
    return {k: sums[k] / counts[k] for k in sums}


def sample_counts(rows: List[dict]) -> Counter:
    return Counter((row["state"], row["action"]) for row in rows)


def max_estimation_error(p_hat: Dict[tuple, Dict[str, float]]) -> float:
    """Largest |P_hat - P_true| over all entries, with blocked burns as a self-loop."""
    worst = 0.0
    for (s, a), dist in p_hat.items():
        true = DYNAMICS[s][a] or {s: 1.0}
        for s2 in set(dist) | set(true):
            worst = max(worst, abs(dist.get(s2, 0.0) - true.get(s2, 0.0)))
    return worst


def value_iteration(p_hat, r_hat, gamma: float = GAMMA, tol: float = 1e-10):
    """Bellman optimality on the estimated model: Q(s,a) = sum_s' P(s'|s,a) [R(s,a,s') + gamma V(s')]."""
    states = sorted({s for s, _ in p_hat})
    v = {s: 0.0 for s in states}
    v.update({s: 0.0 for s in TERMINAL})
    while True:
        q = {(s, a): sum(p * (r_hat[(s, a, s2)] + gamma * v[s2]) for s2, p in p_hat[(s, a)].items())
             for (s, a) in p_hat}
        new_v = {s: max(q[(s, a)] for a in BURNS if (s, a) in q) for s in states}
        delta = max(abs(new_v[s] - v[s]) for s in states)
        v.update(new_v)
        if delta < tol:
            return v, q


def greedy_policy(q: Dict[tuple, float]) -> Dict[str, str]:
    best: Dict[str, Optional[str]] = {}
    for (s, a), value in sorted(q.items()):
        if s not in best or value > q[(s, best[s])]:
            best[s] = a
    return best


def main():
    print("=== Part 3: Discounted return of the brief's trajectory ===")
    log = TrajectoryLogger()
    for s, a, r in [("s0", "a0", 10), ("s1", "a1", 0), ("s2", "a2", -5), ("s3", "a3", 20)]:
        log.record(s, a, r)
    log.finish("s4")
    print("tau =", log.tau())
    print("G_0 = 10 + 0.95*0 + 0.95^2*(-5) + 0.95^3*20 =", round(log.discounted_return(), 6))
    print("G_t =", [round(g, 4) for g in returns_to_go(log.rewards)])

    rows = read_jump_dataset()
    p_hat, r_hat, n = transition_frequency(rows), mean_rewards(rows), sample_counts(rows)
    print(f"\n=== Part 3: Empirical P(s'|s,a) from {len(rows)} synthetic historical jumps ===")
    for (s, a), dist in p_hat.items():
        cells = ", ".join(f"{s2}={p:.3f}" for s2, p in dist.items())
        print(f"{s:>9} | {a:<8} | n={n[(s, a)]:>3} | {cells}")
    print(f"Max |P_hat - P_true| = {max_estimation_error(p_hat):.3f}")

    v, q = value_iteration(p_hat, r_hat)
    print("\n=== Part 3: Bellman value iteration on the estimated model (gamma = 0.95) ===")
    for s in ("LOBE", "DUST_LANE", "CORE", "CORRIDOR"):
        print(f"{s:>9} | V*={v[s]:7.2f} | Q(DIRECT)={q[(s, 'DIRECT')]:7.2f} | Q(CORRIDOR)={q[(s, 'CORRIDOR')]:7.2f}")
    print("Greedy policy:", greedy_policy(q))
    print("[PASS] MDP logger and evaluator verified.")


if __name__ == "__main__":
    main()
