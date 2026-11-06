# Stato del manoscritto e workflow dei risultati

`paper.tex`, `sections/` e `references.bib` sono le fonti LaTeX autorevoli. La bozza aggiornata descrive: il problema sperimentale; la letteratura su parallel PSO, topologie e island models; le cinque modalità di esecuzione; le sette topologie implementate; la campagna architetturale principale a tre topologie; l'espansione complementare a sette topologie; dieci baseline PSO; COCO BBOB, BBOB-LargeScale e CEC 2013 LSGO.

La formulazione matematica CPSO è considerata verificata e fissata nella fase di sviluppo del modello. Il presente manoscritto non riapre tale verifica: si concentra sulle semantiche degli esecutori, sul disegno sperimentale e sulla valutazione empirica. Non sono ancora inseriti risultati confirmatory, discussione empirica o conclusioni: le relative sezioni verranno completate dopo l'esecuzione dei codici e la generazione dei risultati. Nessun valore di pilot o smoke test va presentato come evidenza confirmatory.

## Sequenza operativa

Il piano dettagliato di comandi per il MacBook Air M2 è in [`MACBOOK_AIR_M2_RUNBOOK.md`](MACBOOK_AIR_M2_RUNBOOK.md). Le campagne già definite sono:

- `configs/campaigns/cpso_serial_parallel_pilot.json` — equivalenza sincrona e scaling (1.400 run);
- `configs/campaigns/cpso_dynamics_pilot.json` — asincronia e isole (720 run);
- `configs/campaigns/cpso_topology_expansion_pilot.json` — sette topologie (1.050 run);
- `configs/benchmarks/bbob_stageC_pilot.json` — COCO BBOB (770 run);
- `configs/benchmarks/bbob_largescale_preconfirmatory.json` — BBOB-LargeScale pre-confirmatory (1.056 run);
- `configs/benchmarks/cec2013_stageC_pilot.json` — CEC 2013 LSGO (198 run).

Le directory di output suggerite sono parte delle convenzioni del post-processing; non sovrascriverle con campagne differenti. I runner salvano manifest e JSONL riprendibili.

## Figure

Le figure interne si generano con:

```bash
python analysis/postprocess_publication_figures.py \
  --results-root results \
  --output paper/figures/internal_pilots
```

I riepiloghi descrittivi dei benchmark esterni si generano con:

```bash
python analysis/postprocess_external_benchmarks.py \
  --results-root results/external_benchmarks \
  --output paper/figures/external_pilots
```

Per COCO, le ECDF e i grafici runtime-to-target ufficiali devono essere generati con `cocopp` sui dati Observer prodotti in `coco_data/<method>/`; i riepiloghi per rango finale sono supplementari. Non aggregare risultati con suite, budget, hardware o versioni software differenti.

## Compilazione LaTeX

Dalla directory `paper/`, con `pdflatex` e BibTeX installati:

```bash
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
bibtex paper
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
```

`paper_pre_revision.pdf` è un artefatto storico non sincronizzato e non deve essere usato. `paper.pdf` va rigenerato dalle fonti sopra dopo ogni modifica.

## Regola di interpretazione

Il confronto esterno attuale include CPSO e dieci PSO discrete, non CMA-ES o differential evolution: non consente affermazioni di superiorità sull'intero stato dell'arte degli ottimizzatori continui. Le modalità sincrone parallele possono essere confrontate in termini di speedup solo dopo avere verificato l'equivalenza di traiettoria. Asincronia e isole cambiano la dinamica e vanno confrontate a FE budget fissato, non chiamate accelerazioni equivalenti del seriale.
