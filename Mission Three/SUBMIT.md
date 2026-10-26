# Submission steps

1. Each officer reviews their part and ticks their sign-off box in `mission-03-doc.md` (and the matching boxes at the end of `04_Memory_Hierarchy_and_Safety_Audit.md` and `ps03_solution.ipynb`). Then rebuild the PDF with `python build_doc.py`.
2. Commit and push the Mission Three folder with the commit message required by the brief:

```bash
cd /path/to/aura-fei-fei
git status --short
git add "Mission Three/" README.md
git -c core.pager=cat diff --cached --stat
git commit -m "feat(ps03): restore AURA Subsystem 03 Spatial Attention Mesh (SEC-03-M45)"
git push origin main
```

3. Check that the public Mission Three link opens, including the notebook, the PDF and `evidence/`.
4. Upload `mission-03-doc.pdf` to the Moodle assignment drop-box.
5. On https://dlearn-cai.web.app/ go to Mission 3, open 4. Transmit Solution, paste https://github.com/bilal-younus/aura-fei-fei and complete the four station authorizations.

The brief's suggested path `docs/weeks/week-03/problem_set/` is not used; this repository keeps one folder per mission, as for Missions 1 and 2. Confirm with the instructor if the exact path is required. Do not commit `.venv/`; it is ignored.
