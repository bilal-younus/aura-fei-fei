"""
Shared Eta Carinae / Homunculus Nebula environment for Mission 06.

The sector graph, transition probabilities and rewards are a synthetic, seeded teaching model.
They are not course data or real astrophysics; every number is defined in this file.
"""

import csv
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

LOBE, DUST_LANE, CORE, CORRIDOR = "LOBE", "DUST_LANE", "CORE", "CORRIDOR"
OPEN_SPACE, HULL_BREACH = "OPEN_SPACE", "HULL_BREACH"
SECTORS = (LOBE, DUST_LANE, CORE, CORRIDOR, OPEN_SPACE, HULL_BREACH)
TERMINAL = (OPEN_SPACE, HULL_BREACH)

# Two warp headings. DIRECT points straight at open space (the greedy System 1 reflex);
# CORRIDOR is the clear gravitational corridor reported by the sensor array.
HEADINGS = {"DIRECT": (1.0, 0.0, 0.0), "CORRIDOR": (-0.85, 0.52, 0.05)}
BURNS = tuple(HEADINGS)
SHIELD_SAFE = 60          # minimum shield % for burns that start in the radiation of the core

# True dynamics P(s' | s, burn) with nominal shields (>= SHIELD_SAFE).
# DUST_LANE + DIRECT is blocked by a gravitational shear wall: the jump is aborted and nothing changes.
DYNAMICS = {
    LOBE: {"DIRECT": {CORE: 0.8, DUST_LANE: 0.2}, "CORRIDOR": {CORRIDOR: 0.85, DUST_LANE: 0.15}},
    DUST_LANE: {"DIRECT": None, "CORRIDOR": {CORRIDOR: 0.9, DUST_LANE: 0.1}},
    CORE: {"DIRECT": {OPEN_SPACE: 0.5, CORE: 0.5}, "CORRIDOR": {CORRIDOR: 0.7, CORE: 0.3}},
    CORRIDOR: {"DIRECT": {OPEN_SPACE: 0.9, CORRIDOR: 0.1}, "CORRIDOR": {OPEN_SPACE: 0.95, CORRIDOR: 0.05}},
}
# Burning out of the core with shields below SHIELD_SAFE: radiation breaches the hull with this probability.
LOW_SHIELD_BREACH = 0.4

REWARDS = dict(burn=-1.0, shields=-0.5, scan=0.0, tool_error=-5.0, OPEN_SPACE=100.0, HULL_BREACH=-100.0)


def burn_reward(aborted: bool, next_sector: str) -> float:
    """Reward of one burn: fuel cost, or the error penalty if aborted, plus the terminal outcome."""
    if aborted:
        return REWARDS["tool_error"]
    return REWARDS["burn"] + REWARDS.get(next_sector, 0.0)


@dataclass
class ShipState:
    sector: str = LOBE
    shields: int = 30           # % plasma shield intensity
    plasma: int = 100           # remaining plasma reserve (fuel)
    hull: int = 100

    def snapshot(self) -> dict:
        return asdict(self)


def heading_of(x: float, y: float, z: float):
    """Name of the known heading within cosine 0.95 of (x, y, z), or None."""
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0:
        return None
    for name, (a, b, c) in HEADINGS.items():
        cos = (x * a + y * b + z * c) / (norm * math.sqrt(a * a + b * b + c * c))
        if cos >= 0.95:
            return name
    return None


class NebulaEnvironment:
    """Stochastic warp dynamics; all randomness comes from one seeded generator."""

    def __init__(self, start: str = LOBE, shields: int = 30, seed: int = 0):
        self.ship = ShipState(sector=start, shields=shields)
        self.rng = random.Random(seed)

    def sample(self, distribution: dict) -> str:
        r, total = self.rng.random(), 0.0
        for sector, p in distribution.items():
            total += p
            if r < total:
                return sector
        return sector

    def warp(self, burn: str) -> str:
        """Apply one burn. Raises ValueError when the jump is impossible (ship state unchanged)."""
        s = self.ship
        if s.sector in TERMINAL:
            raise ValueError(f"ship is in terminal sector {s.sector}; no further jumps")
        distribution = DYNAMICS[s.sector][burn]
        if distribution is None:
            raise ValueError(f"gravitational shear wall blocks the {burn} heading from {s.sector}; jump aborted")
        if s.sector == CORE and s.shields < SHIELD_SAFE and self.rng.random() < LOW_SHIELD_BREACH:
            nxt = HULL_BREACH
        else:
            nxt = self.sample(distribution)
        s.plasma -= 5
        s.hull = 0 if nxt == HULL_BREACH else s.hull
        s.sector = nxt
        return nxt


def generate_jump_dataset(n: int = 600, seed: int = 606) -> list:
    """Synthetic 'historical jump' log: uniform random sector and burn, nominal shields, true dynamics."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        start, burn = rng.choice((LOBE, DUST_LANE, CORE, CORRIDOR)), rng.choice(BURNS)
        env = NebulaEnvironment(start=start, shields=SHIELD_SAFE, seed=rng.randrange(2**31))
        try:
            nxt, aborted = env.warp(burn), False
        except ValueError:
            nxt, aborted = start, True      # blocked jump: the ship stays where it was
        rows.append(dict(jump_id=i, state=start, action=burn, next_state=nxt,
                         aborted=int(aborted), reward=burn_reward(aborted, nxt)))
    return rows


def write_jump_dataset(path: Path = ROOT / "data" / "historical_jumps.csv") -> Path:
    rows = generate_jump_dataset()
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_jump_dataset(path: Path = ROOT / "data" / "historical_jumps.csv") -> list:
    with path.open() as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    print("Wrote", write_jump_dataset().relative_to(ROOT))
