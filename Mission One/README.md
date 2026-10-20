# AURA Mission 01 - Squadron Fei-Fei

## Team roles
- Bilal Younus: Core Architect and submission coordinator.
- Danish Hafeez: Protocol & Interface Specialist.
- Anas Arshad: Knowledge & Memory Archivist and Security & Ethical Navigator.

## Run
Python 3.10 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_securely.py
```

The launcher defaults to Gemini 3.1 Flash-Lite and prompts for a key without saving or displaying it. Alternatively configure GEMINI_API_KEY. Gemini generation uses Google's native REST API. The openai package supports the optional OpenAI provider. No OpenAI account is needed for Gemini mode.

Set AURA_MODEL to select another supported model. Optional OpenAI mode uses AURA_PROVIDER=openai and OPENAI_API_KEY. Do not commit secrets. Free-tier quota and eligibility depend on the Google account; running the script does not change billing settings.

A successful run saves 05_Execution_Transcript.txt and 06_API_Run_Evidence.json. A placeholder transcript is not execution evidence.

## Offline tests
`python -m unittest discover -p test_engine.py`

10 tests cover control flow and response checks using synthetic fixtures. They do not substitute for live execution.

## Deliverables
The written brief, protocol map, ethics sandbox, Python engine, evaluation worksheet and report builder are included. START_HERE.md lists submission checks. Live evidence and its qualitative evaluation are complete. Team review and authorization precede submission.
