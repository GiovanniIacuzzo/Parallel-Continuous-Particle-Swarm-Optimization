# Studio III del CPSO parallelo

La campagna parte dal commit `3baa956cd28390905aa8b453e4bc28478878fad0`. Dalla radice del repository, applicare la patch dello ZIP e ricompilare la libreria nativa.

```bash
git apply --check /percorso/CPSO_StudioIII.patch
git apply /percorso/CPSO_StudioIII.patch
source .venv/bin/activate
python -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
python -m pytest -q tests/python/test_study3_budgets.py
```

Se `git apply --check` segnala un conflitto, usare un worktree nuovo basato sul commit indicato. La patch è pensata per il repository, che contiene già l'API adattiva asincrona. Lo script `study3.py` usa una nuova API sincrona. Se `CPSO_SERIAL_LIBRARY` indica una libreria precedente, aggiornare il percorso alla nuova build o rimuovere l'override.

## Verifica rapida sul Mac

Eseguire le campagne in sequenza, mantenendo un solo thread interno per i pacchetti numerici. Usare Python e build nativi arm64.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
python study3.py screen --profile smoke
python study3.py ablation --profile smoke
python study3.py fe --profile smoke
python study3.py time --profile smoke
```

Per verificare anche BBOB, installare la dipendenza del repository e lanciare il suo smoke.

```bash
python -m pip install -r requirements-benchmarks-coco.txt
python study3.py fe --suite bbob --profile smoke
```

I risultati dello smoke sono in `results/study3_smoke/`. Ogni fase completata produce `complete.json`, CSV in `analysis/` e PDF vettoriali in `figures/`.

## Campagna pilota

Prima eseguire screening e ablazione. Il secondo comando usa la ricetta congelata dal primo.

```bash
python study3.py screen
python study3.py ablation
```

Lo screening confronta **otto ricette sincrone**, incrociando casualità scalare o per coordinata, vicinato Global o Random-8 e passo 0.38125 o 1. I fattori sono confrontati con stessi seed, inizializzazione, budget e numero di worker. L'ablazione usa altri semi e incrocia passo fisso/adattivo e recovery disabilitato/abilitato sulla ricetta vincente. Entrambe le campagne usano 50.000 e 500.000 FE, popolazione 100, Rastrigin, Rosenbrock e Ackley a (D=20,100) e cinque semi per cella. I due budget sono esecuzioni indipendenti. Lo screening esegue 480 run e l'ablazione 240.

Lo screening sceglie il rango medio più basso sui blocchi completi. A parità esatta, favorisce la ricetta di riferimento. L'ablazione sceglie un adattamento solo se il guadagno medio rispetto al passo fisso raggiunge 0.05 unità di log10 del gap. La soglia è esplorativa e serve a congelare una scelta prima della validazione. Dopo 12 generazioni senza un nuovo best, il passo adattivo cresce del 25% fino a tre volte il valore di partenza. Quando il best migliora, il passo si riduce del 15%, con limite inferiore pari al valore di partenza. Il recovery sostituisce l'aggiornamento del quinto meno competitivo dello sciame, preservando il leader globale. Il candidato è perturbato intorno al leader sociale con raggio per coordinata pari a due deviazioni standard dello sciame, limitato all'1–20% del box. Ogni candidato usa una delle FE della generazione. Il CSV riporta il numero di attivazioni, candidati di recovery, candidati che battono il best della generazione precedente e dispersione media ai trigger.

### Confronto a FE fissate

```bash
python study3.py fe
python study3.py fe --suite bbob
```

Il primo comando confronta la ricetta congelata con dieci varianti PSO, un controllo discreto con la stessa geometria e gli eventuali controlli CPSO fissi. Usa 50.000 e 500.000 FE e tre semi per funzione e dimensione. Il secondo usa sei funzioni BBOB trasformate, due istanze a (D=20), e 10.000 e 50.000 FE. L'Observer ufficiale verifica i conteggi e fornisce il minimo di riferimento. Le chiamate all'Observer sono seriali, mentre il CPSO mantiene la propagazione parallela. Su BBOB usare i risultati di qualità in funzione delle FE.

Si può misurare anche il tempo per completare le stesse FE sulla diffusione inversa. Eseguire questa campagna in una directory dedicata e importare la selezione precedente.

```bash
python study3.py fe --functions inverse_diffusion --budgets 2000 10000 \
  --output results/study3_inverse_fe --development results/study3_pilot
```

### Confronto a tempo fissato

```bash
python study3.py time
```

Il confronto usa 5 e 15 secondi, cinque semi, (D=20,100), Rastrigin e diffusione inversa con 512 passi RK4. Misura il CPSO a uno e quattro worker e SPSO-2011, CSO, CLPSO, SL-PSO a quattro worker. In totale i budget assegnati sommano circa 40 minuti. Il controllo a un worker usa lo stesso backend del CPSO parallelo e viene escluso dalla graduatoria competitiva. Il programma confronta le storie best-so-far allineate alle FE fino al prefisso comune. Per una prova iniziale con circa quattro minuti di budget totale, usare invece una campagna separata.

```bash
python study3.py time --seconds 5 --seeds 2 \
  --output results/study3_time_rapido --development results/study3_pilot
```

I metodi condividono la funzione obiettivo C++ e la possibilità di valutare un batch su più worker. Gli aggiornamenti dello sciame usano C++ nel CPSO e Python/NumPy nelle baseline. Il tempo comprende inizializzazione e creazione del pool. Come regola comune, una soluzione diventa disponibile al termine del suo batch di valutazione. Le FE di un batch terminato oltre il limite sono registrate separatamente e il suo best non entra nel risultato. Il tempo effettivo di ritorno può superare la scadenza per finire l'ultimo batch e chiudere il pool. Lo schedule a tempo fissato impiega un orizzonte di 1.000 generazioni e poi mantiene il valore finale. L'orizzonte è indipendente dal tetto di emergenza delle FE. Il raggiungimento del tetto invalida la run a tempo fissato. Per un confronto con otto worker usare `--workers 8` e una directory nuova con la stessa `--development`.

## Conferma indipendente

Dopo aver completato e valutato il pilot, il profilo `confirm` importa la stessa ricetta congelata ed esegue campagne più ampie. Usa altri semi e, per BBOB, istanze e funzioni diverse dal pilot, escluse anche F1, F8, F15 e F21 già usate nel fattoriale storico. La campagna temporale include 15 e 60 secondi e richiede circa dieci ore di budget aggregato con le impostazioni predefinite. Il costo va verificato sul Mac a partire dal pilot.

```bash
python study3.py fe --profile confirm --development results/study3_pilot
python study3.py fe --suite bbob --profile confirm --development results/study3_pilot
python study3.py time --profile confirm --development results/study3_pilot
```

Per riprendere una campagna interrotta, rilanciare lo stesso comando. Il runner valida hash e protocollo e riusa ogni run completata. Usare un nuovo percorso per cambiare parametri o build. Durante le misure usare un Mac alimentato, con impostazioni stabili, ed evitare campagne simultanee.

## Risultati da condividere

Dopo i primi due comandi pilota, inviare `screen/` e `ablation/`. I primi file da leggere sono `selected.json`, `analysis/leaderboard.csv`, `analysis/rank_summary.csv` e `analysis/paired_*.csv`. Per valutare la compatibilità con il pilot precedente conservare anche `protocol.json` e `runs/`.

Dopo i confronti inviare anche `fe/`, `fe_bbob/` e `time/`. Sono di interesse `leaderboard.csv`, `function_wins_cpso_selected.csv`, `best_baseline_function_wins.csv`, `target_hits.csv`, `ert.csv`, `parallel_prefix_checks.csv` nel caso temporale e i PDF. Il conteggio contro la migliore baseline per funzione è descrittivo. I test appaiati sono riportati per baseline, con correzione Holm. I target del runner sono osservati a fine batch. L'ERT somma il costo al primo hit per i successi e il budget consumato o assegnato per le run censurate, quindi divide per il numero di successi.

```bash
zip -qr risultati_studio3_pilot.zip results/study3_pilot
```
