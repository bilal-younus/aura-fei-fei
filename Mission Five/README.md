# Mission Five - Dual-Path Associative Memory

Squadron Fei-Fei, Conversational AI, BTU WS 2026/27.

## Run
Python 3.11+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
python -m unittest -v test_ps05.py
```

The full local benchmark builds 25,000 vectors of dimension 768. All four numbered scripts also run independently. `ps05_solution.ipynb` integrates the scripts and recorded NotebookLM audit; `mission-05-doc.md` is converted to `mission-05-doc.pdf` by `python build_doc.py`.

Neural corpus and fixture embeddings are cached for offline evaluation. Rebuilding them with `python embeddings.py` downloads the public all-MiniLM-L6-v2 model. The cached encoder intentionally rejects unknown text rather than silently substituting a different embedding method. Add new evaluation text to `input_texts()` and rebuild to extend the demonstration.

The official course webpage and repository did not provide a separately named technical-archive extract. Document C is a labelled course-website summary, with provenance recorded in `data/`. Ship serial numbers and maintenance intervals in the local retrieval corpus are explicitly synthetic fixtures; they are not official course specifications.

`notebooklm_audit_report.md` documents actual browser observations, citation judgments, absent-fact probes and an audio excerpt. Replaying Python summarizes that preserved external evidence; it does not rerun the cloud service. No private account credentials or full research PDFs are included in the repository.

The top-level `Mission Five` folder follows the squad repository's mission naming convention. `weeks/week-05/` provides compatibility launchers for the handout's paths. Implementation details and limitations are in the report.
