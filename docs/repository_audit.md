# Audit della repository — CPSO parallelo

**Data:** 6 novembre 2026  
**Versione software:** 0.7.4  
**C ABI nativo:** 7  
**Base esaminata:** archivio `repository(6).zip`, aggiornato con il protocollo sperimentale locale, la revisione del manoscritto e il post-processing esterno.  
**Perimetro di questo audit:** stato software e sperimentale. La formulazione matematica CPSO è considerata verificata e congelata nel progetto; non viene riaperta in questa fase.

## 1. Stato generale

La repository contiene un motore CPSO continuo C++20, un'interfaccia Python `ctypes`, cinque modalità native di esecuzione, sette topologie sociali, dieci baseline PSO a tempo discreto, adapter per COCO BBOB/BBOB-LargeScale e CEC 2013 LSGO, runner per campagne riproducibili e strumenti di analisi/post-processing.

Le cinque modalità native sono:

| Modalità | Semantica | Scopo sperimentale |
|---|---|---|
| `serial` | Generazioni sincrone | Riferimento deterministico |
| `master_slave` | Valutazioni obiettivo parallele, commit sincrono | Efficienza senza cambiare la traiettoria |
| `particle_parallel` | Propagazione/valutazione parallela, commit deterministico | Efficienza senza cambiare la traiettoria |
| `asynchronous` | Aggiornamenti event-driven | Studio di una dinamica asincrona distinta |
| `islands` | Sottopopolazioni con migrazione | Studio di decomposizione e comunicazione |

Le sette topologie implementate sono global-best, ring, von Neumann, wheel, random connected, Watts–Strogatz small-world e Barabási–Albert scale-free. La campagna architetturale principale usa tre topologie canoniche; una campagna distinta estende lo studio alle sette. Questa distinzione è intenzionale e deve essere mantenuta anche nei risultati.

Le dieci baseline PSO sono classic global-best, inertia-weight global-best, constriction-factor, inertia-weight local-best ring, CLPSO, SL-PSO, SPSO-2011, FIPS, Competitive Swarm Optimizer e Dynamic Multi-Swarm PSO. Le campagne esterne confrontano quindi undici metodi in totale, CPSO incluso. Il set è una selezione rappresentativa per famiglie e meccanismi PSO, non una dichiarazione di copertura esaustiva dello stato dell'arte degli ottimizzatori continui.

## 2. Verifiche software eseguite nell'aggiornamento

| Verifica | Esito |
|---|---:|
| Configurazione e build CMake Release | Superata |
| CTest | 1/1 test target superato |
| Suite Python completa | 114/114 test superati |
| Test del post-processore esterno | Inclusi nella suite; superati |
| Dry-run campagna smoke sincrona | 195 run pianificate |
| Dry-run pilot sincrono | 1.400 run pianificate |
| Dry-run pilot asynchronous/islands | 720 run pianificate |
| Dry-run pilot espansione topologica | 1.050 run pianificate |
| Dry-run BBOB Stage C | 770 run pianificate |
| Dry-run BBOB-LargeScale smoke | 132 run pianificate |
| Dry-run BBOB-LargeScale pre-confirmatory | 1.056 run pianificate |
| Dry-run CEC smoke | 11 run pianificate |
| Dry-run CEC Stage C | 198 run pianificate |
| Compilazione LaTeX e bibliografia | Superata; PDF aggiornato generato |
| Controllo riferimenti LaTeX | Nessuna citazione o reference non risolta |
| Controllo visivo PDF | Nessun overflow critico; restano solo warning minori di underfull |

Le campagne di cui sopra sono state validate in modalità dry-run, non eseguite integralmente. L'ambiente di audit è Linux/GCC; la build macOS sul MacBook Air M2 dell'utente deve essere eseguita localmente seguendo `paper/MACBOOK_AIR_M2_RUNBOOK.md`.

I test dei driver COCO/CEC eseguiti nell'ambiente di sviluppo verificano i contratti con fixture simulate. Non costituiscono uno smoke test sulle librerie ufficiali né una valutazione delle funzioni benchmark ufficiali. Nessuna campagna confirmatory completa è dichiarata completata.

## 3. Software implementato

### 3.1 Motore nativo e interfaccia

- Motore C++20 con propagatore analitico CPSO e gestione della popolazione.
- Backend seriale, master–slave, particle-parallel, asynchronous e islands.
- Topologie statiche a sette famiglie.
- RNG derivato per particella e gestione deterministica degli aggiornamenti sincroni.
- ABI C versionata e wrapper Python `ctypes`.
- Callback Python custom objective limitata alla modalità seriale: non viene assunta la thread-safety di callback Python nel motore parallelo.

La formulazione CPSO e le convenzioni operative adottate nel progetto sono considerate già verificate e congelate. Modifiche future al modello dovranno essere trattate come variazioni esplicite, versionate e accompagnate da regression test; non costituiscono un prerequisito del piano sperimentale corrente.

### 3.2 Baseline e suite esterne

Le dieci baseline sono implementazioni Python/NumPy distinte. Parametri, topologie, adattamenti e limiti di interpretazione sono documentati in `docs/stage_c_pso_baselines.md`. La callback esterna resta seriale: BBOB, BBOB-LargeScale e CEC confrontano la qualità a budget FE controllato, non misurano lo speedup dei backend nativi.

I benchmark esterni sono:

- COCO BBOB: benchmark continuous single-objective con profili smoke, pilot, pre-confirmatory e confirmatory;
- COCO BBOB-LargeScale: estensione ad alta dimensionalità, con profili dedicati;
- CEC 2013 LSGO: suite large-scale con profili smoke, pilot e confirmatory.

Per COCO, ECDF e runtime-to-target prodotti da `cocopp` sono l'analisi primaria. I riepiloghi a rango generati dal post-processore del progetto sono supplementari e non li sostituiscono.

### 3.3 Campagne interne

La campagna architetturale definitiva è articolata in quattro stage e prevede 9.360 run complessive:

- Stage 1: confronto delle topologie sincrone canoniche;
- Stage 2: dinamica asynchronous;
- Stage 3: architettura islands;
- Stage 4: strong scaling su carico sintetico `sphere_work`.

Il protocollo complementare di espansione topologica usa sette topologie e ha un pilot da 1.050 run e una fase confirmatory da 3.150 run. I pilot sono già configurati per il Mac e sono descritti nel runbook.

### 3.4 Post-processing

- `analysis/postprocess_publication_figures.py`: figure vettoriali per campagne interne.
- `analysis/postprocess_external_benchmarks.py`: completezza delle campagne esterne, confronto a gruppi problema-seme completi, ranghi medi con gestione dei pari merito, bootstrap per cluster di funzione e figure PDF descrittive.
- `cocopp`: generazione ufficiale di ECDF e runtime-to-target BBOB/BBOB-LargeScale.

Il post-processore esterno non inventa risultati: gruppi incompleti non entrano nei ranghi comparativi; il report di completezza ne conserva traccia. Per BBOB i ranghi finali restano secondari rispetto alla valutazione ufficiale COCO.

## 4. Protocollo locale MacBook Air M2

Il runbook `paper/MACBOOK_AIR_M2_RUNBOOK.md` definisce una sequenza eseguibile senza alterare i profili confirmatory:

1. build Release, CTest, pytest e smoke sincrono;
2. pilot sincrono da 1.400 run;
3. pilot asynchronous/islands da 720 run;
4. espansione a sette topologie da 1.050 run;
5. pilot COCO BBOB da 770 run;
6. pilot CEC 2013 LSGO da 198 run, subordinato a dipendenza e licenza;
7. BBOB-LargeScale pre-confirmatory da 1.056 run, facoltativo e più oneroso.

Sul Mac eseguire una campagna per volta, conservare metadati hardware/software e non modificare configurazioni, semi, popolazioni o budget FE a campagna avviata. I risultati locali sono preliminari ma comparabili all'interno dello stesso protocollo. Quando sarà disponibile l'HPC, ripetere prima le stesse configurazioni e i medesimi identificatori problema/seme; confrontare i tempi solo all'interno della stessa macchina, non in valore assoluto tra Mac e HPC.

## 5. Manoscritto

`paper/paper.tex` e le sezioni incluse descrivono il problema sperimentale, lo stato dell'arte, le cinque modalità, le sette topologie, la campagna principale a tre topologie, l'espansione complementare a sette, le dieci baseline e le tre suite esterne. Sono predisposte le sezioni risultati, discussione e conclusioni secondo le research questions del lavoro.

Il paper non contiene risultati confirmatory inventati. Le figure sono richiamate condizionalmente: verranno incluse quando i corrispondenti PDF saranno prodotti dai post-processori. I risultati pilot, quando disponibili, dovranno essere identificati come esplorativi e distinti dalla conferma HPC.

## 6. Questioni ancora aperte, senza riaprire il modello matematico

1. **Esecuzione dei pilot sul Mac:** build nativa macOS e run effettivi delle campagne locali.
2. **Dipendenze ufficiali:** smoke COCO con `coco-experiment`/`cocopp` e smoke CEC con wrapper/licenza verificati.
3. **Prestazioni:** misure su Mac per carichi di valutazione realistici e, successivamente, strong scaling su HPC; il solo `sphere_work` non basta a sostenere un vantaggio applicativo generale.
4. **Statistica:** interpretare le funzioni come unità/cluster di livello superiore dove previsto; non trattare seed, dimensioni o istanze come repliche indipendenti senza giustificazione. Definire e congelare i confronti prima di esaminare i confirmatory.
5. **Validità del confronto topologico:** le sette reti non sono degree-matched; i risultati identificano differenze tra famiglie e parametri implementati, non l'effetto causale isolato della geometria.
6. **Perimetro delle baseline:** il confronto è PSO-vs-PSO; non comprende CMA-ES, differential evolution o l'intero panorama degli ottimizzatori continui.
7. **Portabilità e rilascio:** verificare macOS e CI remota, aggiungere la licenza solo dopo conferma dei diritti, e versionare repository/configurazioni/dati.
8. **Campagne confirmatory:** non eseguire sul Mac base le campagne definitive complete; la loro esecuzione è prevista sull'HPC.

## 7. Giudizio aggiornato

**Software:** architettura e test automatici sono in uno stato maturo per avviare i pilot sul Mac.

**Disegno sperimentale:** sono presenti protocolli locali e confirmatory separati, con configurazioni, runner, manifest e strumenti di post-processing. I profili locali consentono di controllare la pipeline e ottenere confronti preliminari comparabili, senza sostituire la conferma HPC.

**Paper:** la bozza ora è coerente con il perimetro implementato e predisposta per l'inserimento dei risultati. Il lavoro scientifico da completare è l'esecuzione delle campagne, l'analisi delle evidenze, la discussione dei limiti e la stesura delle conclusioni.
