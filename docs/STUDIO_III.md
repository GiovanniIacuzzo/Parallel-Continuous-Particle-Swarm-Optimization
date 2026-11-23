# Campagna comparativa CPSO per il paper

Estrarre lo ZIP nella radice del repository, sopra la patch con ricerca locale e restart. Ricompilare la libreria nativa e usare lo stesso ambiente Python per tutte le suite.

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-benchmarks-coco.txt
python -m pip install 'setuptools>=65,<81' 'Cython>=3,<4' wheel
python -m pip install -r requirements-benchmarks-cec2013.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
python study3_competition.py paper --profile smoke
python study3_competition.py paper
```

Se `CPSO_SERIAL_LIBRARY` è impostata, deve puntare alla libreria appena compilata. Il comando principale esegue le suite in sequenza. Per riprendere un'esecuzione interrotta, rilanciare lo stesso comando con la stessa build. Ogni replica completata viene riusata. `--paper-suite inverse bbob` permette di eseguire solo una parte della campagna, con lo stesso protocollo e nella stessa directory.

## Disegno sperimentale

La campagna confronta un solo CPSO a quattro worker con le dieci baseline PSO disponibili nello Studio III. Le baseline sono Classic GBest, Inertia GBest, Constriction GBest, Inertia LBest Ring, CLPSO, SL-PSO, SPSO-2011, FIPS Ring, CSO e DMS-PSO. Ogni confronto usa quattro worker configurati per metodo. COCO e CEC espongono obiettivi scalari valutati serialmente dall'adattatore; su queste suite l'asse del confronto è il numero di valutazioni.

| Suite | Protocollo | Run |
| --- | --- | ---: |
| Diffusione inversa | D=20 e 100, 512 passi RK4, 20 semi nuovi, timeout 15 s | 440 |
| COCO BBOB | 24 funzioni, D=40, istanze 31–35, 50.000 valutazioni (FE) | 1.320 |
| Native | Le 6 funzioni native, D=20 e 100, originali, traslate e ruotate, 5 semi, 50.000 FE | 4.620 |
| CEC2013 LSGO | 15 funzioni, D=1000 salvo F13/F14 a D=905, 5 semi, 120.000 FE | 825 |

Il CPSO usa un'unica configurazione a passo 1 e topologia Random-8, con ricerca locale, restart e interpolazione quadratica. Il modello unidimensionale usa il leader e i due punti simmetrici già valutati lungo una coordinata. Si prende in considerazione il vertice soltanto se la curvatura è positiva e il minimo previsto è interno al tratto campionato e migliora il miglior valore già valutato. Al massimo un vertice per batch occupa una delle valutazioni del batch successivo. La leadership cambia in base al valore effettivamente calcolato. I contatori `quadratic_trials` e `quadratic_improvements` registrano rispettivamente le proposte valutate e quelle che battono il leader all'apertura del loro batch.

## Risultati da archiviare

I risultati completi vanno in `results/study3_paper_main/`, con una sottocartella per suite. Ogni suite produce `protocol.json`, `complete.json`, `analysis/`, `figures/` con PDF vettoriali e `runs/` con le tracce. Per il paper, usare:

- `leaderboard.csv` e `outcome_counts.csv` per rango medio ed esiti per funzione contro la migliore baseline.
- `outcomes_by_function.csv` e `scores_by_dimension.csv` per il dettaglio del confronto.
- `time_to_target.csv` nella diffusione inversa per successi, tempo mediano e FE mediane per arrivare a un gap di 1e-8.
- `target_hits.csv` e i PDF delle ECDF per l'andamento al variare del target.

Conservare le tracce `runs/` per le analisi successive. Per un primo riepilogo è sufficiente inviare per ogni suite `protocol.json`, `complete.json` e `analysis/`.
