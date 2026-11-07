# Mission 04 - Ethical Steering Core

Squadron Fei-Fei (Group 13), Conversational AI, BTU WS 2026/27.

Repository: https://github.com/bilal-younus/aura-fei-fei

## Run
Python 3.10+ and PyTorch 2.4+. No API key or downloaded model is needed. The tiny causal transformer trains from scratch using deterministic synthetic examples. The held-out human-labelled data sample is included with its licence and provenance.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
python -m unittest -v test_ps04.py
```

Each requested file also runs independently:

```bash
python 01_DPO_Loss_Core.py
python 02_Implicit_Reward_Evaluator.py
python 03_Length_Bias_and_SimPO.py
python 04_Activation_Steering_and_Audit.py
```

`ps04_solution.ipynb` contains executed outputs. `mission-04-doc.md` is the report source; `python build_doc.py` converts it to `mission-04-doc.pdf`. `evidence/` contains model checkpoints, raw metrics, figures, test output and execution details.

The included checkpoints reproduce the recorded evaluation. For a fresh deterministic training run, move `evidence/checkpoints.pt` aside, then run `python run_all.py`. The seed is 404; exact floating-point values may differ across PyTorch versions and hardware. Training settings and hashes are in `evidence/training.json`. Python package versions are recorded in `evidence/environment-lock.txt`.

## Measurement boundaries
- Calibration uses 64 actual human chosen/rejected labels from the Anthropic HH-RLHF helpful-base test split, never used for training. The tiny synthetic-trained model is not a capable pretrained assistant; low validation performance is retained.
- Likelihoods score final 160 prompt bytes and first 96 response bytes. Byte tokens are not words. Truncation flags are recorded per pair.
- The measured length statistic uses scored token counts; it is not a causal estimate of generated verbosity.
- The 50% saving is exact model-parameter storage (two copies versus one), not an invented total-VRAM measurement. The script profiles peak CUDA allocation when CUDA exists; this host has no CUDA GPU.
- Steering results are closed-candidate likelihood selections on eight synthetic prompts per task, not free-form safety certification. Calibration and held-out steering prompts are separate. Weights remain unchanged and hooks are removed after every coefficient.

## Submission
Upload `mission-04-doc.pdf`, and provide the public repository URL through the course portal and Moodle as requested. The notebook is also available. Four team members review their assigned sections and authorize their own roles in the portal. Local execution evidence does not certify IDE use or personal sign-offs.
