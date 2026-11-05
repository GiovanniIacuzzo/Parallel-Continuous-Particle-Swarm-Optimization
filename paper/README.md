# Stato del manoscritto

`paper.tex`, `sections/` e `references.bib` sono le fonti LaTeX autorevoli.
`paper.pdf` è la bozza corrente generata da queste fonti: contiene abstract,
introduzione, modello e semantica delle architetture, protocollo sperimentale e
bibliografia. Il PDF è di lavoro, non pronto per la submission: mancano i
risultati della campagna definitiva, la discussione empirica, le conclusioni e
la verifica integrale della fedeltà al testo sorgente.

`paper_pre_revision.pdf` è un artefatto della versione precedente e non è
sincronizzato con il sorgente attuale. Non deve essere citato o inviato come
versione aggiornata del manoscritto.

Per rigenerare il PDF dalla directory `paper/` in un ambiente TeX con
`pdflatex` e `bibtex8`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
bibtex8 paper
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
```

La compilazione con `latexmk` è possibile se l'ambiente ha un eseguibile BibTeX
funzionante e correttamente configurato.
