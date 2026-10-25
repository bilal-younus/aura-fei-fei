# Submission steps

1. Review the PDF and assigned roles.
2. VS Code execution is recorded in evidence/VSCode_Run.md. The model evaluation used the Claude browser interface; Antigravity use and portal approvals are not certified by this evidence.
3. Push this entire Mission Two folder to GitHub using the commands below.
4. Check that the public Mission Two link opens, including the PDF, code, transcript and evidence.
5. Upload `Mission02_Report.pdf` to the Mission 02 submission on Moodle if that is your course's upload entry. Paste https://github.com/bilal-younus/aura-fei-fei into its URL/text field.
6. Complete the AURA Operations Console submission and station authorizations required by ps02.pdf. The three members cover four roles. Confirm any requirement for four distinct officers and the course-repository relationship with the instructor.

```bash
cd "$HOME/Downloads/Conversational AI"
git status --short
git add "Mission Two/" README.md
git -c core.pager=cat diff --cached --stat
git commit -m "Add Mission Two FST simulator, evaluation and report"
git push origin main
```

Do not rerun git init or reset. Existing Mission One files remain in their folder. No keys are required or included. The code demonstrates navigation; it does not operate any real system.
