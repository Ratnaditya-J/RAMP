# RAMP ArXiv Paper Draft

This directory contains an arXiv-style LaTeX paper draft for the RAMP research package.

Files:

- `ramp_v0_arxiv.tex`: main paper source
- `references.bib`: bibliography
- `figures/*.png`: generated paper figures

Regenerate figures from current local artifacts:

```bash
cd /Users/ratnaditya/RAMP
.venv/bin/python scripts/build_paper_assets.py
```

Compile on a machine with TeX Live/MacTeX or in Overleaf:

```bash
cd /Users/ratnaditya/RAMP/paper
pdflatex -interaction=nonstopmode ramp_v0_arxiv.tex
bibtex ramp_v0_arxiv
pdflatex -interaction=nonstopmode ramp_v0_arxiv.tex
pdflatex -interaction=nonstopmode ramp_v0_arxiv.tex
```

The checked-in PDF can be regenerated with the commands above. LaTeX auxiliary build products are
ignored by Git.
