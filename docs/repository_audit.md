# Repository audit — CPSO parallel project

**Data:** 5 novembre 2026  
**Stato:** sviluppo correttivo successivo all'audit iniziale  
**Base:** archivio `repository(5).zip`; il file originale non è stato modificato.

## 1. Risultato generale

La repository ora dispone di un nucleo C++20 compilabile, wrapper Python `ctypes`, cinque modalità di esecuzione, campagne configurabili, analisi statistica con unità gerarchiche esplicite, checkpoint recuperabili, provenienza hardware/software e una bozza LaTeX riallineata al codice.

La pulizia e le correzioni implementative hanno rimosso alcune incoerenze, ma non equivalgono alla chiusura scientifica del progetto. In particolare, la fedeltà del modello seriale al testo integrale della formulazione sorgente, l'esecuzione della campagna definitiva, l'aggiunta di benchmark esterni e la scelta della licenza richiedono ancora input o risorse non presenti nell'archivio.

## 2. Verifiche effettivamente eseguite

Le verifiche locali sono state condotte su Linux x86_64, GNU C++ 14.2.0, build CMake Release, Python e Matplotlib 3.10.8.

| Verifica | Esito |
|---|---:|
| Configurazione e build CMake Release | Superata |
| CTest | 1/1 test target superato |
| Suite Python | 38/38 test superati |
| `compileall` su Python, esperimenti e test | Superato |
| Smoke qualità e dinamiche | 12/12 run |
| Smoke performance `sphere_work` | 15/15 run |
| Campagna smoke complessiva | 27/27 run |
| Validazione FE e copertura degli smoke study | Superata |
| Generazione CSV, report e figure dello smoke | Superata |
| `--dry-run` campagna definitiva | 9.360 run conteggiate, nessuna esecuzione |
| Compilazione LaTeX con BibTeX8 | Superata; PDF di lavoro di 5 pagine |

I due smoke study sono test di integrazione, non esperimenti scientifici. Il ramo qualità ridotto usa una sola funzione e tre semi; pertanto non permette alcuna inferenza sulle prestazioni generali o sulla significatività per famiglie di funzione. I relativi risultati e grafici non sono inseriti nel manoscritto come evidenza scientifica.

La build e i test sono stati ripetuti dopo le ultime modifiche; non è stata eseguita in questa fase una nuova matrice completa di sanitizer. Le verifiche sanitizer positive dell'audit precedente non vanno interpretate come una certificazione generale di assenza di race condition.

## 3. Interventi implementativi completati

### 3.1 Modello e semantica documentati

- `docs/mathematical_model.md` descrive l'ODE a coefficienti congelati, il propagatore analitico, il campionamento casuale, le mappe di vincolo, il budget FE e le modalità esecutive.
- `paper/sections/section_2_model_and_semantics.tex` esplicita il modello implementato e le differenze tra aggiornamento sincrono, event-driven e isole.
- Il manoscritto contiene una proposizione con dimostrazione per l'invarianza della traiettoria tra seriale, master--slave e particle-parallel, sotto ipotesi dichiarate: funzione obiettivo deterministica e thread-safe, stream casuali per particella, snapshot sociale congelato, regole identiche e commit deterministico.
- La proposizione riguarda il modello codificato e non dimostra che questo coincida con la formulazione sorgente pubblicata.

### 3.2 Integrità delle campagne e recupero

- Ogni record di run JSONL viene scritto e sincronizzato (`flush`/`fsync`) prima di avviare il lavoro successivo.
- I manifest JSON vengono sostituiti tramite file temporaneo, sincronizzazione e `os.replace` atomico.
- Il lettore JSONL può eliminare in resume soltanto un record finale non terminato e malformato. Record interni malformati, record duplicati o oggetti senza `run_key` generano errore.
- Un ultimo record JSON valido ma privo di newline viene conservato e normalizzato in modalità ripresa.
- La validazione dei risultati controlla esattamente tutte le combinazioni previste di funzione, dimensione, seed e metodo; rifiuta duplicati, combinazioni mancanti/inattese, FE errati, valori non finiti e storie non monotone.
- La validazione `--dry-run` resta non invasiva e non richiede la libreria nativa.

### 3.3 Provenienza e versionamento

Il manifest registra sistema, release, architettura, modello CPU quando disponibile, conteggi CPU logici/fisici, versione Python, dipendenze, variabili d'ambiente di threading e metadati del motore nativo. CMake passa inoltre al C API compilatore, versione compilatore, build type, standard C++, architettura target e flag effettivi.

La versione del progetto e della libreria è `0.6.0`; la versione ABI C resta `5`. Sono numeri distinti e sono verificati dai test Python.

### 3.4 Statistica e unità sperimentali

L'analisi non considera più le dodici combinazioni funzione-dimensione come dodici famiglie indipendenti. Le sei funzioni analitiche sono i cluster di livello superiore; le due dimensioni sono misure ripetute annidate nella funzione e i semi sono repliche appaiate nella cella funzione-dimensione.

- I confronti appaiati sono ridotti prima per dimensione e poi a una differenza mediana per famiglia.
- Il signed-rank bilaterale usa enumerazione esatta delle configurazioni di segno; gli zeri sono esclusi dal ranking.
- La correzione di Holm è applicata ai p-value grezzi e verificata da un test di regressione.
- Il test omnibus di Friedman mantiene la statistica dei ranghi ma calibra il p-value con 20.000 permutazioni dei metodi all'interno di ciascuna famiglia.
- Gli intervalli sono stimati con bootstrap gerarchico famiglia → dimensione → seed appaiati.
- Le dimensioni funzione-dimensione rimangono anche disponibili come riepiloghi descrittivi.

Con sei famiglie la potenza statistica è limitata: il test esatto signed-rank bilaterale non può produrre p-value arbitrariamente piccoli. La scelta di sei funzioni analitiche limita inoltre la generalizzabilità a suite più ampie.

### 3.5 Documentazione e manoscritto

La bozza è stata ristretta a ciò che la repository implementa: cinque modalità, tre topologie sociali, sei funzioni analitiche, dimensioni 30 e 100 e scaling condiviso mediante `sphere_work`. Sono state rimosse le affermazioni non supportate da implementazioni COCO/BBOB, CEC, PSO standard, GPU/MPI o decomposizione cooperativa.

Il PDF attuale è una bozza di lavoro di cinque pagine: abstract, introduzione, modello/semantica e protocollo sperimentale. Non contiene ancora risultati definitivi, discussione empirica o conclusioni; non è un manoscritto pronto per la submission.

## 4. Valutazione del codice e dell'architettura

| Modalità | Comportamento | Relazione con il seriale |
|---|---|---|
| `serial` | Generazioni sincrone | Riferimento |
| `master_slave` | Parallelizza le valutazioni obiettivo | Traiettoria identica richiesta e testata |
| `particle_parallel` | Parallelizza inizializzazione/propagazione e valutazioni | Traiettoria identica richiesta e testata |
| `asynchronous` | Event-driven con pubblicazione immediata | Dinamica diversa; a più worker dipende dall'ordine degli eventi |
| `islands` | Sottopopolazioni e migrazione | Dinamica diversa per struttura dell'informazione |

La distinzione tra accelerazione di un esecutore equivalente e modifica dell'algoritmo è una scelta metodologica da mantenere. Per l'analisi di scaling, `sphere_work` è un carico sintetico di costo controllabile, non un benchmark di ottimizzazione.

La suite locale verifica l'equivalenza per i casi testati e la soluzione dell'ODE contro un oracolo RK4 indipendente e proprietà di semigruppo. Questi test aumentano la fiducia nell'implementazione, ma non dimostrano in generale assenza di race condition o correttezza per ogni callback fornita dall'utente.

## 5. Criticità residue e dipendenze esterne

### A. Fedeltà alla formulazione sorgente — bloccante scientifico

Il modello codificato è ora descritto senza ambiguità, ma non è stata chiusa la verifica equazione-per-equazione rispetto all'articolo integrale di Orlando e Ricciardello. Il materiale pubblico raggiungibile durante questo lavoro espone solo un publisher preview e invita a richiedere il PDF completo agli autori; non consente di controllare tutte le equazioni e convenzioni. Restano da confrontare, in particolare:

- coefficienti casuali scalari rispetto a coordinate o vettori;
- ODE esatta, variabili e convenzioni di inerzia;
- calendario e indicizzazione temporale;
- distribuzioni iniziali di posizioni e velocità;
- regole di vincolo e restituzione;
- istanti di osservazione dell'obiettivo e aggiornamento dei best personali/sociali.

**Input necessario:** PDF integrale o sorgente affidabile dell'articolo. Sino ad allora il paper qualifica l'implementazione come ricostruzione operativa documentata, non come replica rigorosa certificata.

### B. Benchmark esterni e baseline — non implementati

Non ci sono adapter COCO/BBOB, BBOB-Large-Scale o CEC 2013 LSGO, né baseline discrete PSO standard. Non sono implementati backend MPI/GPU o decomposizione cooperativa. Il manoscritto è stato ristretto per non rivendicare questi risultati; per un articolo che sostenga superiorità o generalizzazione su benchmark esterni occorre decidere e implementare una suite e baseline confrontabili con FE, stop, seed e hardware controllati.

### C. Campagna definitiva — non eseguita

La configurazione prevede 9.360 run e 273.888.000 valutazioni. La campagna è stata contata e validata in `--dry-run`; sono state eseguite soltanto campagne smoke da 27 run. Non sono quindi disponibili speedup definitivi, ERT definitivi, tassi di successo, intervalli o evidenza statistica per sostenere le ipotesi di ricerca. La campagna completa va eseguita sull'hardware target e i dati grezzi conservati.

### D. Ampiezza della base di problemi e inferenza

La struttura statistica è stata corretta per non equiparare due dimensioni della stessa funzione a due funzioni indipendenti. Rimane però un limite sostanziale: sei famiglie analitiche sono poche e non coprono la varietà di problemi reali o suite standard. Gli intervalli gerarchici e i test esatti non compensano una base di problemi ristretta.

### E. Piattaforme e CI

La CI GitHub Actions per Ubuntu e macOS è stata aggiunta con matrice Python 3.10/3.12, build CMake, CTest, pytest e compileall. È una configurazione pronta, ma non è stata ancora eseguita sul servizio GitHub. Le verifiche locali effettive restano Linux/GCC. Windows non è incluso; macOS deve ancora essere validato sul Mac target.

### F. Licenza

L'archivio originale non includeva una licenza testuale. La dichiarazione MIT non supportata è stata rimossa da `pyproject.toml` e lo stato è documentato in `docs/license_status.md`. Prima della pubblicazione il titolare deve confermare licenza, titolarità e anno, aggiungendo poi il file completo. Questa scelta non può essere fatta tecnicamente al posto del titolare.

### G. Storia Git e risultati

L'archivio ZIP non include la storia Git. La copia di sviluppo conserva file sorgente, configurazioni e documentazione, ma non può ricostruire commit, branche o cronologia non forniti. È raccomandato importare il risultato in un repository Git, creare un commit iniziale e taggare lo stato `0.6.0` prima di lanciare la campagna definitiva.

## 6. Sequenza di sviluppo ancora necessaria

1. Ricevere il PDF integrale della formulazione sorgente e chiudere la verifica matematica; aggiungere test dove emergano differenze.
2. Decidere il perimetro editoriale: studio architetturale circoscritto oppure confronto esterno con COCO/BBOB e PSO standard. Implementare adapter/baseline solo se richiesti dalla domanda scientifica.
3. Eseguire la CI GitHub e la build sul Mac di destinazione; includere eventuali correzioni platform-specific.
4. Confermare licenza e aggiungere il testo legale coerente.
5. Eseguire la campagna definitiva (9.360 run) su hardware dichiarato, mantenendo i JSONL grezzi, manifest e log.
6. Validare completezza, FE, equivalenza, qualità, incertezza e scaling; solo dopo completare risultati e conclusioni del paper.

## 7. Giudizio aggiornato

**Implementazione e infrastruttura:** buono stato di sviluppo, con test, equivalenza sincrona esplicitata, contabilità FE, checkpoint recuperabili e analisi gerarchica più prudente.

**Validazione scientifica:** ancora aperta. La fedeltà al modello sorgente e la campagna definitiva sono prerequisiti non negoziabili per conclusioni forti. La bozza ora descrive correttamente la portata attuale, ma non è pronta per una submission.
