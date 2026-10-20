"""
AURA Dialogic Engine — Mission 1
Implements a Socratic refinement loop and a three-persona Galilean debate.

Setup: Python 3.10+, pip install -r requirements.txt
Run: python run_securely.py (defaults to Gemini; key is prompted without saving).
Environment: AURA_PROVIDER=gemini, GEMINI_API_KEY, AURA_MODEL=gemini-3.1-flash-lite.
Optional OpenAI: AURA_PROVIDER=openai and OPENAI_API_KEY.
Keys must never be committed. Provider quota and billing depend on your account.
"""

import os
import json
import urllib.request
import urllib.error
import time
import hashlib
import sys
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


PROVIDER = os.getenv("AURA_PROVIDER", "gemini")
MODEL = os.getenv("AURA_MODEL", "gemini-3.1-flash-lite" if PROVIDER == "gemini" else "gpt-4.1-mini")
MAX_ITERATIONS = 3


@dataclass
class SocraticTurn:
    iteration: int
    premise: str
    response: str


@dataclass
class SocraticResult:
    status: str
    final_premise: str
    turns: List[SocraticTurn] = field(default_factory=list)


class AuraEngine:
    def __init__(self, model: str = MODEL, client=None):
        if client is not None:
            self.client = client
            self.model = model
            self.calls = []
            return
        if OpenAI is None:
            raise RuntimeError(
                "OpenAI package is missing. Install it with: pip install openai"
            )
        key_name = "GEMINI_API_KEY" if PROVIDER == "gemini" else "OPENAI_API_KEY"
        if not os.getenv(key_name):
            raise RuntimeError(f"{key_name} is not configured.")
        options = {"api_key": os.environ[key_name], "timeout": 90.0, "max_retries": 0}
        if PROVIDER == "gemini":
            options["base_url"] = "https://generativelanguage.googleapis.com/v1beta/openai/"
        self.client = OpenAI(**options)
        self.calls = []
        self.model = model

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the configured language model."""
        if PROVIDER == "gemini" and OpenAI is not None and isinstance(self.client, OpenAI):
            return self.ask_gemini(system_prompt, user_prompt)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )
        choice = response.choices[0]
        if choice.finish_reason != "stop" or getattr(choice.message, "refusal", None):
            raise RuntimeError("Model response was incomplete or refused; rerun after review.")
        content = (choice.message.content or "").strip()
        if not content:
            raise RuntimeError("Model returned an empty response.")
        self.calls.append({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "requested_model": self.model,
            "returned_model": getattr(response, "model", self.model),
            "system_prompt": system_prompt, "user_prompt": user_prompt,
            "output": content,
        })
        return content

    def ask_gemini(self, system_prompt: str, user_prompt: str) -> str:
        system_prompt += "\nKeep the response under 180 words while retaining all required labels."
        cache_dir = Path(__file__).with_name("run_cache")
        cache_dir.mkdir(exist_ok=True)
        digest = hashlib.sha256(json.dumps([self.model, system_prompt, user_prompt]).encode()).hexdigest()
        cache_file = cache_dir / (digest + ".json")
        if cache_file.exists():
            record = json.loads(cache_file.read_text(encoding="utf-8"))
            self.calls.append(record)
            return record["output"]
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1536,
                                 "thinkingConfig": {"thinkingLevel": "low"}},
        }
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    data = json.load(response)
                break
            except urllib.error.HTTPError as error:
                print(f"Google API HTTP {error.code}")
                if error.code not in (429, 500, 502, 503) or attempt == 2:
                    raise
                time.sleep(35)

            except (TimeoutError, urllib.error.URLError):
                if attempt == 2:
                    raise
                print("Connection timeout; retrying saved request.")
                time.sleep(5)

        candidates = data.get("candidates", [])
        if not candidates or candidates[0].get("finishReason") != "STOP":
            raise RuntimeError("Gemini response incomplete or blocked.")
        content = "\n".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", [])
                            if not p.get("thought")).strip()
        if not content:
            raise RuntimeError("Gemini returned an empty response.")
        self.calls.append({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                           "requested_model": self.model, "returned_model": data.get("modelVersion"),
                           "system_prompt": system_prompt, "user_prompt": user_prompt,
                           "output": content, "usage": data.get("usageMetadata", {})})
        cache_file.write_text(json.dumps(self.calls[-1], indent=2, ensure_ascii=False), encoding="utf-8")
        Path(__file__).with_name("06_Partial_API_Evidence.json").write_text(
            json.dumps(self.calls, indent=2, ensure_ascii=False), encoding="utf-8")
        return content

    def socratic_loop(self, initial_premise: str) -> SocraticResult:
        """Refine a premise through questions, counterexamples, and revision."""
        current = initial_premise.strip()
        if not current:
            raise ValueError("Premise must not be empty.")
        history: List[SocraticTurn] = []

        system_prompt = """
You are AURA, a careful Socratic discovery assistant.
For each iteration:
1. Clarify important terms and assumptions.
2. Ask focused probing questions.
3. Give one relevant counterexample.
4. Say whether it reveals a contradiction, limitation, or ambiguity.
5. Propose a refined premise if useful.
Be respectful. Do not invent evidence. If unresolved, say APORIA.
Return these labels exactly: QUESTIONS, COUNTEREXAMPLE, TEST, REFINED PREMISE, STATUS.
STATUS must be REFINE, RESOLVED_WITH_CAVEATS, or APORIA.
A claim surviving a few tests is not proven universally true.
"""
        status = "aporia"

        for iteration in range(1, MAX_ITERATIONS + 1):
            prompt = (
                f"Iteration {iteration} of {MAX_ITERATIONS}.\n"
                f"Current premise: {current}\n"
                "Previous turns: " + json.dumps([t.response for t in history]) + "\n"
                "Test this premise and return the required labels. Use plain LABEL: text, no Markdown."
            )
            answer = self.ask(system_prompt, prompt)
            history.append(SocraticTurn(iteration, current, answer))
            print(f"\n--- SOCRATIC ITERATION {iteration} ---")
            print(answer)

            # The model's response is text, so parse the requested labels cautiously.
            refined = self._extract_section(answer, "REFINED PREMISE")
            raw_status = self._extract_section(answer, "STATUS").upper()

            if raw_status not in {"REFINE", "RESOLVED_WITH_CAVEATS", "APORIA"}:
                raise ValueError("Missing or invalid STATUS in model response.")
            if not all(self._extract_section(answer, x) for x in
                       ("QUESTIONS", "COUNTEREXAMPLE", "TEST", "REFINED PREMISE")):
                raise ValueError("Missing required response section.")
            if raw_status == "RESOLVED_WITH_CAVEATS":
                status = "resolved_with_caveats"
                if refined and refined.lower() not in {"none", "n/a"}:
                    current = refined
                break
            if raw_status == "APORIA":
                status = "aporia"
                if refined and refined.lower() not in {"none", "n/a"}:
                    current = refined
                break
            if (refined and refined.lower() not in {"none", "n/a"}
                    and refined.casefold() != current.casefold()):
                current = refined
                status = "refining"
            else:
                status = "aporia"
                break

        if status == "refining":
            status = "aporia"  # Iteration budget exhausted; no terminal resolution.
        return SocraticResult(status=status, final_premise=current, turns=history)

    @staticmethod
    def _extract_section(text: str, label: str) -> str:
        """Extract text after a label until the next expected uppercase label."""
        labels = [
            "QUESTIONS", "COUNTEREXAMPLE", "TEST",
            "REFINED PREMISE", "STATUS"
        ]
        start_marker = label + ":"
        start = text.upper().find(start_marker)
        if start < 0:
            return ""
        start += len(start_marker)
        end = len(text)
        for other in labels:
            if other == label:
                continue
            position = text.upper().find(other + ":", start)
            if position >= 0:
                end = min(end, position)
        return text[start:end].strip(" \n:-")

    def galilean_debate(self, topic: str) -> str:
        """Run Advocate, Traditionalist, and Mediator turns."""
        advocate_prompt = """
You are Salviati, the Advocate. Make the strongest reasonable case for the
proposal. Identify benefits, assumptions, and what evidence would test your view.
Do not invent facts; label uncertainty.
"""
        traditionalist_prompt = """
You are Simplicio, the Traditionalist/Critic. Challenge the proposal fairly.
Identify risks, missing evidence, and unintended consequences. Do not assume
the proposal is necessarily wrong. Label uncertainty and avoid invented facts.
"""
        mediator_prompt = """
You are Sagredo, the Mediator. Compare the two arguments, identify agreements
and disagreements, assess evidence quality, and give a conditional recommendation.
Do not automatically split the difference. State unresolved questions.
"""

        advocate = self.ask(advocate_prompt, f"Debate topic: {topic}")
        print("\n--- SALVIATI: ADVOCATE ---\n", advocate)

        traditionalist = self.ask(
            traditionalist_prompt,
            f"Debate topic: {topic}\n\nSalviati's argument:\n{advocate}\n"
            "Respond critically and fairly."
        )
        print("\n--- SIMPLICIO: TRADITIONALIST ---\n", traditionalist)

        rebuttal = self.ask(
            advocate_prompt,
            f"Debate topic: {topic}\nYour opening: {advocate}\n"
            f"Critic: {traditionalist}\nRespond to the strongest objection and revise if warranted."
        )
        print("\n--- SALVIATI: REBUTTAL ---\n", rebuttal)

        mediator = self.ask(
            mediator_prompt,
            f"Debate topic: {topic}\n\n"
            f"Salviati's argument:\n{advocate}\n\n"
            f"Simplicio's response:\n{traditionalist}\n\n"
            f"Salviati rebuttal:\n{rebuttal}\n\nSynthesize the debate."
        )
        print("\n--- SAGREDO: MEDIATOR ---\n", mediator)

        return (
            "SALVIATI — ADVOCATE\n" + advocate +
            "\n\nSIMPLICIO — TRADITIONALIST\n" + traditionalist +
            "\n\nSALVIATI — REBUTTAL\n" + rebuttal +
            "\n\nSAGREDO — MEDIATOR\n" + mediator
        )


def main() -> None:
    print("=" * 68)
    print("AURA DIALOGIC ENGINE — MISSION 1")
    print("=" * 68)

    try:
        aura = AuraEngine()
    except RuntimeError as error:
        print(f"\nSetup error: {error}")
        print("Follow the setup instructions at the top of this file.")
        raise SystemExit(1)

    premise = input(
        "\nEnter a premise for Socratic testing "
        "(press Enter for the example): "
    ).strip()
    if not premise:
        premise = "An AI agent is benevolent if it obeys human commands."

    socratic_result = aura.socratic_loop(premise)
    print("\n=== SOCRATIC SUMMARY ===")
    print("Status:", socratic_result.status)
    print("Final/best premise:", socratic_result.final_premise)

    topic = input(
        "\nEnter a debate topic (press Enter for the example): "
    ).strip()
    if not topic:
        topic = ("Should frontier AI models be open-sourced? A university lab has a model "
                 "useful for education and coding, with unresolved cyber-misuse concerns. "
                 "Compare immediate weights/code release, staged evaluation access, and "
                 "restricted access. Copies cannot reliably be recalled. Capability evidence "
                 "is incomplete; do not invent evaluation results.")

    debate_text = aura.galilean_debate(topic)

    transcript = [
        "AURA DIALOGIC ENGINE — EXECUTION TRANSCRIPT",
        f"Model: {aura.model}",
        f"Run UTC: {datetime.now(timezone.utc).isoformat()}",
        "Source: genuine API responses; interrupted runs may resume from cached responses",
        f"Initial premise: {premise}",
        "",
        "SOCRATIC TURNS:",
    ]
    for turn in socratic_result.turns:
        transcript.append(f"\nIteration {turn.iteration}")
        transcript.append(f"Premise: {turn.premise}")
        transcript.append(turn.response)
    transcript.extend([
        "\nSOCRATIC SUMMARY",
        f"Status: {socratic_result.status}",
        f"Final/best premise: {socratic_result.final_premise}",
        "\nGALILEAN DEBATE",
        debate_text,
    ])

    output_dir = Path(__file__).resolve().parent
    (output_dir / "06_API_Run_Evidence.json").write_text(
        json.dumps(aura.calls, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "05_Execution_Transcript.txt").open("w", encoding="utf-8") as file:
        file.write("\n".join(transcript))

    print("\nSaved transcript to 05_Execution_Transcript.txt")
    print("Review the transcript and verify the model outputs before submission.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Execution failed ({type(error).__name__}). No completed transcript was generated.", file=sys.stderr)
        raise SystemExit(1)
