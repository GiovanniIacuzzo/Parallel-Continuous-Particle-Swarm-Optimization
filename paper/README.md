# Manoscritto: stato e workflow

`paper.tex`, `sections/`, `appendices/` e `references.bib` sono le fonti LaTeX autorevoli. Il manoscritto è stato riallineato al codice attuale e distingue il modello matematico, le semantiche dei cinque esecutori, il protocollo sperimentale, i risultati da compilare e i dettagli supplementari.

## Convenzione terminologica

Nel paper **CPSO** identifica esclusivamente la *Continuous Particle Swarm Optimization* di Orlando e Ricciardello (2021). Non viene usato per il Cooperative PSO, per altre formulazioni continue precedenti o per le dieci baseline discrete. Le baseline sono nominate con il proprio nome: CLPSO, SL-PSO, SPSO-2011, FIPS, Competitive Swarm Optimizer, DMS-PSO e le varianti classiche.

## Stato scientifico dei risultati

La versione corrente è una bozza di protocollo e manoscritto, non un articolo con risultati confirmatory completi. Le celle `TBD` e i riquadri "Figura da generare" sono segnaposto intenzionali. I pilot e gli smoke test presenti nel repository non vengono presentati come risultati confermativi. Non sono formulate conclusioni di superiorità.

La sezione principale dei risultati contiene i riepiloghi per RQ1–RQ5. L'Appendice E raccoglie i breakdown per funzione/dimensione, statistiche topologiche, migrazioni, decomposizione temporale e sensibilità a `dt`, così da mantenere il corpo dell'articolo leggibile. L'Appendice A rende esplicito il propagatore; B documenta le baseline; C riporta la proprietà condizionale dello span dovuta alla casualità scalare; D documenta le configurazioni e la contabilità FE.

## Configurazioni da cui partire

- `configs/campaigns/definitive_architecture_campaign.json`: campagna interna architetturale, profilo corrente con `dt=0.35`.
- `configs/campaigns/cpso_topology_expansion_pilot.json`: espansione esplorativa delle sette topologie.
- `configs/campaigns/cpso_dt_sensitivity_m2_screening.json`: screening esplorativo del passo temporale su MacBook Air M2; il campo `reference_dt=30` è solo un riferimento del grafico, non il passo congelato delle campagne.
- `configs/benchmarks/bbob_confirmatory.json`, `bbob_largescale_confirmatory.json` e `cec2013_confirmatory.json`: profili esterni pianificati. Non avviare le campagne complete prima del pilot di costo e della verifica del budget.

Per i dettagli sui vincoli statistici, i conteggi FE e le decisioni aperte, vedere `../CODE_AUDIT.md` e l'Appendice D.

## Figure definitive

I percorsi previsti sono elencati in [`figures/results/README.md`](figures/results/README.md). Le figure devono essere preferibilmente PDF vettoriali. Non sovrascrivere i pilot in `figures/internal_pilots/`, `figures/atlas/` o `figures/dt_m2_screening/`.

## Compilazione

Dalla directory `paper/`, con `pdflatex`, `latexmk` e `bibtex8` installati:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -e '$bibtex = "bibtex8 %O %B";' paper.tex
```

`paper.pdf` viene generato dalle fonti sopra. `paper_pre_revision.pdf` è un artefatto storico e non è sincronizzato con la revisione corrente.
