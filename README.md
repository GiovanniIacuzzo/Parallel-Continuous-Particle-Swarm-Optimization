# Parallel Continuous Particle Swarm Optimization

Studio del CPSO di Orlando e Ricciardello su Apple Silicon. Il punto di ingresso è `study.py`. L'unità di confronto per la qualità è la valutazione di funzione (FE). Per lo scaling è il tempo misurato dal motore C++ sul medesimo lavoro.

## Installazione sul Mac M2

Aprire un terminale nella directory che contiene questo README. Servono Python 3.10–3.12 (verifica eseguita con 3.12), CMake ≥3.20 e un compilatore C++20. Se mancano i Command Line Tools di Apple, installarli con `xcode-select --install`. Usare un terminale e un interprete arm64. `python3 -c 'import platform; print(platform.machine())'` deve stampare `arm64` per misurare la build nativa M2.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install cmake 'setuptools>=65,<81' wheel 'Cython>=3,<4'
python -m pip install -e '.[analysis,test]'
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
python -m pytest -q
python tests/test_publication_scaling.py
```

Per le suite esterne installare anche le seguenti dipendenze nella stessa venv. Il pacchetto CEC 2.2 importa Cython durante la build e usa `pkg_resources` a runtime. Il compilatore di Apple è sufficiente sul Mac.

```bash
python -m pip install -r requirements-benchmarks-coco.txt
python -m pip install --no-build-isolation -r requirements-benchmarks-cec2013.txt
```

## Prima esecuzione

```bash
python study.py dt
python study.py scaling
python study.py dynamics
python study.py bbob
python study.py largescale
python study.py cec
```

Il profilo predefinito è `pilot`. I comandi usano automaticamente il candidato congelato prima della validazione. La prima analisi usa 5 candidati, 5 seed appaiati, 6 funzioni, D=30 e 100, 30 iterazioni. Il controllo indipendente riutilizza fino a 4 valori di dt su 30 nuovi seed. Per lo scaling vengono eseguiti 576 run per obiettivo (2 seed e 2 ripetizioni). La dinamica richiede 480 run. Le due suite COCO usano 132 run ciascuna, CEC ne usa 88. Gli intervalli del pilot servono a verificare il flusso e valutare il costo della campagna più ampia.

## Esperimenti per il manoscritto

Ripetere i sei comandi aggiungendo `--profile paper` a ciascuno, iniziando da `python study.py dt --profile paper`. Ogni profilo crea i propri dati, configurazioni e figure. I due profili usano seed distinti nella selezione e nella validazione di dt. I problemi esterni del pilot si sovrappongono al protocollo esteso. Fissare quest'ultimo prima di interpretare le classifiche.

| Comando | Disegno del profilo paper | Risultato principale |
|---|---|---|
| dt | 16 intervalli, inclusi 0.38125, 3.6 e 30, sei obiettivi, D=30/100, N=64, 500 iterazioni, 30 seed. 5.760 run di selezione, poi 30 seed indipendenti sui primi tre candidati e il riferimento 30 | Profilo globale e per funzione, confronto appaiato del candidato congelato sul holdout |
| scaling | Rastrigin e diffusione inversa (2048 passi RK4), D=30/100, 8 grafi, 1/2/4/8 worker, 10 seed, 5 ripetizioni. 14.400 esecuzioni native in totale | Tempi, speedup, efficienza e PDF per obiettivo, grafo e dimensione |
| dynamics | Sei funzioni, D=30/100, 12 seed, 20 configurazioni di seriale, asincrono e isole. 2.880 run | Qualità finale e tempo per variante a budget FE fissato |
| bbob | 24 funzioni, D=5/20, 5 istanze, budget 400D per metodo | 2.640 run, ranghi appaiati e log COCO Observer |
| largescale | 24 funzioni, D=20/80, 5 istanze, budget 80D per metodo | 2.640 run, ranghi e log Observer |
| cec | F1–F15, D=1000 (905 per F13/F14), 5 seed, checkpoint 400/2.000/4.000 FE | 825 run e ranghi a ogni checkpoint |

Per COCO e CEC il profilo paper è deliberatamente **a budget ridotto** per il portatile. In particolare CEC caratterizza soltanto le prime 4.000 FE, molto prima delle 3 milioni del protocollo completo. Le configurazioni di riferimento integrali sono conservate in `configs/benchmarks/*_confirmatory.json`. I parametri effettivi, compreso dt, vengono scritti per ciascun esperimento in `results/`. Le dieci baseline sono PSO classico gbest, inerziale gbest, a costrizione gbest, inerziale lbest ring, CLPSO, SL-PSO, SPSO-2011, FIPS, CSO e DMS.

## Lettura dei risultati

| Percorso relativo a results/pilot oppure results/paper | Uso |
|---|---|
| 01_dt/selection/analysis e 01_dt/holdout/analysis | `selection_summary.json`, ranking, profili per problema, confronti holdout e PDF. I JSONL grezzi sono nelle directory sorelle `raw/`. |
| 02_parallel/{rastrigin,inverse_diffusion} | `publication_scaling_summary.csv`, `publication_scaling_paired.csv`, misure JSONL, manifest della build e PDF vettoriali. |
| 02_parallel/dynamics/raw | `summary.csv`, `paired_summary.csv`, run individuali e manifest. |
| 03_comparison/{bbob,largescale,cec}/analysis | `external_rank_by_problem.csv`, `external_rank_summary.csv`, `external_campaign_completeness.csv` e PDF. I log ufficiali COCO sono in `raw/exdata/`. |

**Scelta del framework.** Confrontare `runtime_seconds` e `serial_speedup` allo stesso obiettivo, D, grafo e numero di worker. Il tempo assoluto individua il backend più rapido. `strong_speedup` usa come riferimento lo stesso backend a un worker e misura quanto scala. L'efficienza è il rispettivo speedup diviso per il numero di worker. Il confronto sincrono verifica uguaglianza esatta di posizione migliore, valore, storia e conteggio FE rispetto al seriale. Le otto configurazioni rappresentano sette famiglie di grafo poiché random è misurata con grado 4 e 8. L'asincrono e le isole hanno dinamiche diverse, confrontate nell'esperimento dedicato. Le isole supportano solo topologie interne compatibili, la griglia è esclusa in quel backend.

**Scelta delle affermazioni.** I ranghi esterni assegnano lo stesso peso a ciascuna funzione e un rango medio agli ex aequo. Il CSV per problema conserva funzione, dimensione, seed, valore e rango di ogni metodo. Verificare che il conteggio atteso coincida con i run unici e che i gruppi incompleti siano zero. Per affermare che il CPSO prevale sulla maggioranza delle funzioni, riportare quante funzioni vince, con quale budget e rispetto a quali metodi, distinguendo parità e sconfitte. I ranghi medi da soli non rispondono a questa domanda. Le callback esterne eseguono CPSO seriale nativo e baseline Python. L'evidenza sull'accelerazione riguarda gli obiettivi nativi misurati nello scaling. Un vantaggio di qualità a pari tempo sui benchmark esterni richiederebbe un esperimento ulteriore.

**Ripresa.** Rilanciare lo stesso comando per dt, scaling, dynamics e CEC. La fase BBOB conserva le tracce complete dell'Observer e richiede una directory grezza nuova dopo un'interruzione. Per esempio, spostare `results/pilot/03_comparison/bbob/raw` in un percorso di archivio dentro `results/`, quindi rilanciare `python study.py bbob`. Lo stesso vale per largescale. Per rigenerare soltanto i ranking da run completati, ad esempio:

```bash
python analysis/postprocess_external_benchmarks.py --results-root results/pilot/03_comparison/bbob/raw --output results/pilot/03_comparison/bbob/analysis
```

Per una misura riproducibile del Mac, eseguire le fasi in sequenza e annotare modalità di alimentazione e carico della macchina. Per condividere l'intero pilot eseguire `zip -qr risultati_m2_pilot.zip results/pilot`.

## Audit della riorganizzazione

Base GitHub `3433f4de2cbd2de024444849027d6f2c0899d99f`. Il codice di ottimizzazione e l'ABI C sono quelli della revisione di partenza. CMake conserva le asserzioni nei due test che le usano anche in Release. Il wrapper ctypes carica il binario compilato localmente da `build/lib`. L'archivio contiene sorgenti portabili.

La revisione di partenza aveva runner di scaling e post-processing sovrapposti, generatori di atlanti, script di pulizia e configurazioni storiche. La distribuzione conserva un solo percorso documentato per ogni domanda sperimentale e le fixture dei test ancora applicabili. Il numero di file Python, inclusi wrapper e test, scende da 53 a 31. In origine lo scaling usava 0.38125 come default, i confronti esterni usavano 0.35 e il runner di scaling misurava soltanto la diffusione inversa. Il flusso principale propaga ora il medesimo dt a tutte le fasi e mette a confronto un obiettivo economico e uno computazionalmente più oneroso. I parametri nei template storici vengono sostituiti prima dell'esecuzione. Il manoscritto in `paper/` è la copia integrale della revisione di partenza e può quindi descrivere campagne precedenti. I dati delle esecuzioni locali dell'autore erano esclusi da Git.
