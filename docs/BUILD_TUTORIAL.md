# Build the SOZLab Tutorial PDF

This guide explains how to regenerate `docs/tutorial.pdf` from `docs/tutorial.md`.

## Prerequisites
- Python 3.11+ recommended (Python 3.10+ should work)
- A local virtual environment for build tooling

## Steps
1) Create and activate a virtual environment (once):
```bash
python3 -m venv .venv_tutorial
. .venv_tutorial/bin/activate
```

2) Install ReportLab (once per venv):
```bash
pip install reportlab
```

3) Build the PDF (preferred, explicit interpreter):
```bash
.venv_tutorial/bin/python scripts/build_tutorial_pdf.py
```

## Output
- Markdown source: `docs/tutorial.md`
- Generated PDF: `docs/tutorial.pdf`

## Notes
- The build script uses ReportLab to render headings, lists, and code blocks.
- Code blocks are wrapped to avoid overflowing page margins.
- The script uses a multi-pass build so the Table of Contents includes page numbers.
