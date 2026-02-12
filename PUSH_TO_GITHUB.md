# PUSH PLAN: Publish SOZLab to GitHub

## Pre-flight checklist (local)
- [ ] Review the repo for sensitive data (tokens, private keys, machine-specific secrets).
- [ ] Confirm `CITATION.cff` metadata is final.
- [ ] Verify `docs/tutorial.pdf` is up to date (if you keep the PDF in-repo).
- [ ] Ensure local outputs ignored by `.gitignore` are not tracked.

## Git initialization and first push
```bash
git init
git add .
git commit -m "Release prep: docs and repo hygiene"

# Set default branch
git branch -M main

# Add remote (replace with your repo URL)
git remote add origin https://github.com/aleff-ferreira/sozlab.git

git push -u origin main
```

## Tag and release
```bash
# Create a tag for the release
git tag v0.1.1

git push origin v0.1.1
```

## GitHub UI checklist
- [ ] Enable secret scanning and push protection.
- [ ] Enable Dependabot alerts.
- [ ] Add repository topics (e.g., gromacs, mdanalysis, solvent).
- [ ] Create a GitHub Release for `v0.1.1` and paste notes from `CHANGELOG_TUTORIAL.md`.
