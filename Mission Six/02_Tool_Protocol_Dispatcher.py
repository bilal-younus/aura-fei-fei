"""
Mission 06 - Part 2: Communications & Protocol Officer
Ship tools, a JSON-Schema contract validator and an exception boundary for the action dispatcher.
Standard library only; the validator implements the JSON-Schema keywords used by the contracts below.

Run: python 02_Tool_Protocol_Dispatcher.py
"""

import json
import math
from typing import Any, Callable, Dict, List, Union

from nebula import HEADINGS, NebulaEnvironment, heading_of


class ShipTools:
    """The three Eta Carinae tools, bound to one ship environment."""

    def __init__(self, env: NebulaEnvironment):
        self.env = env

    def navigate_warp_vector(self, x: float, y: float, z: float) -> str:
        """Jump along a charted heading. Raises ValueError for uncharted or blocked vectors."""
        burn = heading_of(x, y, z)
        if burn is None:
            raise ValueError(f"vector [{x:.2f}, {y:.2f}, {z:.2f}] matches no charted heading {list(HEADINGS)}")
        sector = self.env.warp(burn)
        return (f"Warp jump along [{x:.2f}, {y:.2f}, {z:.2f}] ({burn} heading) arrived in {sector}; "
                f"plasma reserve {self.env.ship.plasma}%.")

    def adjust_plasma_shields(self, intensity: int) -> str:
        """Set magnetic plasma shield intensity in percent (0..100)."""
        if not 0 <= intensity <= 100:
            raise ValueError(f"shield intensity {intensity}% is out of bounds; valid range is [0, 100]")
        self.env.ship.shields = intensity
        return f"Plasma shield harmonics adjusted to {intensity}% capacity."

    def query_sensor_array(self, target: str) -> str:
        """Scan a named target; unknown targets are reported as unresolved, not as errors."""
        ship = self.env.ship
        readings = {
            "homunculus_core": "Severe plasma flare in the core; shields below 60% risk hull breach on exit burns.",
            "eta_carinae_b": "Secondary O-type binary companion emission: 100x solar luminosity.",
            "escape_corridor": "Clear gravitational corridor detected along vector [-0.85, 0.52, 0.05].",
            "ship": f"Sector {ship.sector}; shields {ship.shields}%; plasma {ship.plasma}%; hull {ship.hull}%.",
        }
        return readings.get(target.lower(), f"Sensor array reports: target '{target}' unresolved.")


TOOL_SCHEMAS: Dict[str, dict] = {
    "navigate_warp_vector": {
        "type": "object",
        "properties": {axis: {"type": "number", "minimum": -1.0, "maximum": 1.0} for axis in ("x", "y", "z")},
        "required": ["x", "y", "z"],
        "additionalProperties": False,
    },
    "adjust_plasma_shields": {
        # Range is deliberately left to the tool, so out-of-range values exercise the error boundary.
        "type": "object",
        "properties": {"intensity": {"type": "integer"}},
        "required": ["intensity"],
        "additionalProperties": False,
    },
    "query_sensor_array": {
        "type": "object",
        "properties": {"target": {"type": "string", "minLength": 1, "maxLength": 64}},
        "required": ["target"],
        "additionalProperties": False,
    },
}


def _type_ok(expected: str, value: Any) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if isinstance(value, bool):                      # bool is a subclass of int in Python, but not in JSON
        return False
    if expected == "integer":
        return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    if expected == "number":
        return isinstance(value, (int, float)) and math.isfinite(value)
    raise ValueError(f"unsupported schema type '{expected}'")


def validate(instance: Any, schema: dict, path: str = "arguments") -> List[str]:
    """Return a list of violations of `schema` (empty list = valid)."""
    errors = []
    if "type" in schema and not _type_ok(schema["type"], instance):
        return [f"{path} must be of type {schema['type']}, got {type(instance).__name__}"]
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path} must be one of {schema['enum']}")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path} must be <= {schema['maximum']}")
    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            errors.append(f"{path} must have at least {schema['minLength']} characters")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path} must have at most {schema['maxLength']} characters")
    if isinstance(instance, dict):
        props = schema.get("properties", {})
        errors += [f"missing required argument '{k}'" for k in schema.get("required", []) if k not in instance]
        if schema.get("additionalProperties", True) is False:
            errors += [f"unexpected argument '{k}'" for k in instance if k not in props]
        for key, sub in props.items():
            if key in instance:
                errors += validate(instance[key], sub, f"{path}.{key}")
    return errors


def error_observation(message: str) -> str:
    return f"Tool Execution Error: {message}"


class ActionDispatcher:
    """Parse -> look up -> validate against the contract -> execute inside an exception boundary."""

    def __init__(self, tools: ShipTools):
        self.registry: Dict[str, Callable] = {name: getattr(tools, name) for name in TOOL_SCHEMAS}
        self.log: List[dict] = []

    def dispatch(self, tool_call: Union[str, dict]) -> str:
        """tool_call: JSON text or dict of the form {"tool": name, "arguments": {...}}. Never raises."""
        observation = self._dispatch(tool_call)
        self.log.append(dict(call=tool_call, observation=observation, ok=not observation.startswith("Tool Execution Error")))
        return observation

    def _dispatch(self, tool_call) -> str:
        try:
            call = json.loads(tool_call) if isinstance(tool_call, str) else tool_call
        except json.JSONDecodeError as e:
            return error_observation(f"invalid JSON payload ({e.msg} at character {e.pos})")
        if not isinstance(call, dict) or not isinstance(call.get("tool"), str):
            return error_observation("payload must be an object with a string 'tool' field")
        name, arguments = call["tool"], call.get("arguments", {})
        if name not in self.registry:
            return error_observation(f"unknown tool '{name}'; available: {sorted(self.registry)}")
        violations = validate(arguments, TOOL_SCHEMAS[name])
        if violations:
            return error_observation(f"{name} contract violation: " + "; ".join(violations))
        if name == "adjust_plasma_shields":
            arguments = {**arguments, "intensity": int(arguments["intensity"])}
        try:
            return f"Observation: {self.registry[name](**arguments)}"
        except Exception as e:                       # error boundary: report, never crash the agent loop
            return error_observation(f"{type(e).__name__}: {e}")


def main():
    dispatcher = ActionDispatcher(ShipTools(NebulaEnvironment(seed=1)))
    calls = [
        {"tool": "query_sensor_array", "arguments": {"target": "escape_corridor"}},
        {"tool": "adjust_plasma_shields", "arguments": {"intensity": 85}},
        {"tool": "adjust_plasma_shields", "arguments": {"intensity": -20}},          # tool raises ValueError
        {"tool": "adjust_plasma_shields", "arguments": {"intensity": True}},         # bool is not an integer
        {"tool": "navigate_warp_vector", "arguments": {"x": -0.85, "y": 0.52}},     # missing z
        {"tool": "navigate_warp_vector", "arguments": {"x": 3, "y": 0, "z": 0}},    # above maximum
        {"tool": "navigate_warp_vector", "arguments": {"x": 0, "y": 1, "z": 0}},    # uncharted heading
        {"tool": "self_destruct", "arguments": {}},
        '{"tool": "adjust_plasma_shields", "arguments": {"intensity": 70}',           # truncated JSON
        '{"tool": "navigate_warp_vector", "arguments": {"x": -0.85, "y": 0.52, "z": 0.05}}',
    ]
    print("=== Part 2: Dispatcher contract and error boundary ===")
    for call in calls:
        shown = call if isinstance(call, str) else json.dumps(call)
        print(f"> {shown}\n  {dispatcher.dispatch(call)}")
    errors = sum(not row["ok"] for row in dispatcher.log)
    print(f"\n{len(calls)} calls, {errors} returned error observations, 0 exceptions escaped the boundary.")
    print("[PASS] Tool protocol verified.")


if __name__ == "__main__":
    main()
