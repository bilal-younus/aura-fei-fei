# Mission 02 - FST, Lovelace and noisy channels

Squadron Fei-Fei (Group 13), Conversational AI, BTU WS 2026/27.

Repository: https://github.com/bilal-younus/aura-fei-fei
Mission folder: https://github.com/bilal-younus/aura-fei-fei/tree/main/Mission%20Two

## Run
Python 3.10+; no external dependency or API key for simulator/tests:

```bash
python3 04_fst_noisy_simulator.py
python3 -m unittest -v test_simulator.py
python3 04_fst_noisy_simulator.py --interactive
```

The first command regenerates the ten-turn transcript, five stress traces, channel experiment and local evidence JSON. It does not regenerate the recorded Claude responses or website screenshots. Those are actual external observations, separate from the deterministic simulator.

Optional PDF rebuild: install `requirements.txt`, then `python3 build_report.py`.

## Contents
- `00_Team_Roles.md`: assigned responsibilities.
- `01_Lovelace_Inquiry_Brief.md`: five questions and evaluation.
- `02_FST_and_Noisy_Protocol.md`: states, bounded slots and channel specification.
- `03_Entropy_Hazard_Audit.md`: stress results, entropy/MI metrics and limitations.
- `04_fst_noisy_simulator.py`, `fst_manager.py`, `noisy_channel.py`: runnable code.
- `05_Noisy_Interrogation_Transcript.md`: exactly ten speaker turns.
- `06_Website_Test.md`: successful and failing reference-workbench tests.
- `evidence/`: verbatim model response, raw measurements, test log and screenshots.
- `Mission02_Report.pdf`: report for upload, including GitHub links.

## Recorded workflow
Bilal opened Mission Two in VS Code and ran the simulator and unit tests in its integrated terminal on 25 October 2026. The results were ten compliant turns, five passing stress scenarios and twelve passing tests. The screenshot is in `evidence/vscode_execution.png`.

The five-question model evaluation used the Claude browser interface. Antigravity use and station approvals are not evidenced by this run. Submission instructions are in SUBMIT.md.

Tools: Codex assisted with implementation, tests and report preparation. The recorded evaluation response is from Claude Sonnet 5.5 as labeled in the interface.
