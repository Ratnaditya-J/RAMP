# RAMP v0 arXiv Paper Draft

This directory contains an arXiv-style LaTeX paper draft for the RAMP research-v0 package.

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
pdflatex ramp_v0_arxiv.tex
bibtex ramp_v0_arxiv
pdflatex ramp_v0_arxiv.tex
pdflatex ramp_v0_arxiv.tex
```

The local development machine used for this draft did not have `pdflatex` installed, so the source
and figures were prepared and sanity-checked, but the PDF was not compiled locally.
