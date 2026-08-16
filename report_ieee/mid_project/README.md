# DreamEEG - Midterm Project Report (IEEE format)

LaTeX source for the midterm report, built in the official **IEEEtran**
conference (two-column) format.

## Contents

```
mid_project/
├── main.tex          # the full report (self-contained)
├── figures/
│   ├── image1.png    # non-invasive EEG acquisition illustration
│   └── image2.png    # visual-imagery decoding concept figure
├── main.pdf          # compiled output (4 pages)
└── README.md
```

## How to build on Overleaf

1. Zip the whole `mid_project` folder (make sure `main.tex` and the
   `figures/` folder are inside the zip).
2. In Overleaf: **New Project -> Upload Project** and select the zip.
3. Set the main document to `main.tex` and set the compiler to **pdfLaTeX**.
4. Click **Recompile**. Run it twice if the reference/label numbers look off
   on the first pass.

## How to build locally

```bash
pdflatex main.tex
pdflatex main.tex     # second pass resolves cross-references
```

## Notes

- The report uses the standard `IEEEtran` document class. Overleaf ships it
  by default, so no extra installation is needed.
- The bibliography is **embedded** with `thebibliography`, so there is no
  separate `.bib` file and no BibTeX/biber step to configure.
- The model architecture diagram (Fig. 3) is drawn natively in **TikZ**, so
  it stays crisp at any zoom level and needs no external image.
- Content is based on the DreamEEG midterm proposal
  (`DreamEEG_Midterm_Proposal_v2.pptx`).
