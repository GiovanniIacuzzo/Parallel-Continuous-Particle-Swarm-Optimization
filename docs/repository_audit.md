# Repository audit — CPSO parallel project

**Data:** 5 novembre 2026  
**Versione di sviluppo:** 0.7.1  
**C ABI nativo:** 6  
**Base:** repository derivata dall'archivio `repository(5).zip`; l'archivio originale non è stato modificato.

## 1. Stato generale

La repository contiene un motore C++20 per la CPSO continua, interfaccia Python `ctypes`, cinque modalità di esecuzione su memoria condivisa, tre topologie sociali, campagne interne riproducibili e un livello separato per benchmark esterni e baseline PSO a tempo discreto in spazio continuo.

Il livello esterno comprende adapter COCO `bbob`, COCO `bbob-largescale` e CEC 2013 LSGO; sei baseline NumPy (quattro varianti classiche, CLPSO e SL-PSO); driver per smoke, pre-confirmatory e confirmatory; contabilità FE; checkpoint JSONL; manifest di provenienza e resume vincolato da hash di configurazione, codice, build e dipendenze.

L'implementazione esterna è esclusivamente seriale. Il callback C ABI 6 non presume che i wrapper Python/Cython siano thread-safe e non può essere impiegato con master--slave, particle-parallel, asynchronous o islands. Le campagne esterne comparano qualità a budget FE fissato; non misurano lo scaling delle architetture native.

## 2. Verifiche effettivamente eseguite in questo sviluppo

Ambiente locale: Linux x86_64, GNU C++ 14.2, CMake Release, Python 3.13, NumPy e Matplotlib installati.

| Verifica | Esito |
|---|---:|
| Configurazione e build CMake Release | Superata |
| CTest | 1/1 test target superato |
| Suite Python completa | 83/83 test superati |
| `compileall` su package, esperimenti e test | Superato |
| Custom objective C ABI: contabilità FE e propagazione eccezioni | Superato |
| Sei varianti PSO: determinismo, FE esatti e storie monotone | Superato |
| Driver COCO simulato: esecuzione, FE parity, JSONL e resume | Superato |
| Driver CEC simulato in 1000D: run, checkpoint FE e resume | Superato |
| Preflight COCO: dimensioni, selettori, conteggi e divisibilità FE | Superato |
| Dry-run di tutte le configurazioni COCO/CEC | Superato |
| Dry-run non invasivi | Superato |
| Campagna CPSO seriale/sincrona smoke | 195/195 run completate; equivalenza esatta verificata |
| Smoke di tutte le architetture (seriale, asynchronous, islands e sync parallel) | 27/27 run completate |
| Pilot asynchronous/islands | Dry-run valido: 720 run pianificate |
| Integrazione PSO/CEC-style in D=1000 | 7 metodi, 400 FE ciascuno; superata |
| Compilazione LaTeX + BibTeX | Superata; PDF di lavoro di 7 pagine |
| Rendering e controllo visivo del PDF | Superato |

**Limite della verifica:** i test driver end-to-end usano oggetti COCO/CEC simulati controllati. In questo ambiente offline non sono stati installati `coco-experiment`, `cocopp` o il wrapper CEC; nessun problema ufficiale BBOB/BBOB-LargeScale/CEC è stato valutato. Non sono quindi disponibili risultati esterni e non si afferma di aver completato smoke test sulle librerie ufficiali.

La suite sanitizer non è stata rieseguita dopo l'aggiunta del livello benchmark; i controlli ASan/UBSan e ThreadSanitizer precedenti riguardavano il nucleo già presente e non equivalgono a una certificazione generale di assenza di race condition.

## 3. Componenti aggiunti e modificati

### 3.1 Callback obiettivo seriale e ABI 6

- Aggiunto `CpsoObjectiveCallback` e `cpso_run_custom_serial` al C ABI.
- La funzione C++ richiama il callback in modo sincrono durante la valutazione della particella.
- Il wrapper Python converte il vettore in una sequenza immutabile, intercetta le eccezioni e le ripropaga dopo il ritorno dal C ABI; nessuna eccezione Python attraversa il confine C.
- Le callback personalizzate sono rifiutate per ogni modalità diversa da `serial` e non possono essere combinate con `sphere_work`.
- Il progetto è versionato `0.7.1`, il C ABI `6`; la versione nativa è generata dalla variabile CMake del progetto.

### 3.2 Baseline PSO

Sono implementate sei varianti indipendenti dal motore CPSO:

1. `classic_gbest`: PSO global-best classica, `c1=c2=2`, `w=1`, velocity cap `0.2` della larghezza del dominio.
2. `inertia_gbest`: global-best con `c1=c2=2.05`, inerzia lineare `0.9 -> 0.4` e stesso velocity cap.
3. `constriction_gbest`: global-best con `c1=c2=2.05` e fattore di Clerc, senza velocity cap aggiuntivo.
4. `inertia_lbest_ring`: inerzia lineare e informazione locale sul miglior personale in un anello periodico di raggio uno.
5. `clpso`: Comprehensive Learning PSO, con un exemplar per coordinata, probabilità di apprendimento dipendente dalla particella e refresh dopo sette generazioni senza miglioramento del personal best.
6. `slpso`: Social Learning PSO, con ordinamento worst-to-best, dimostratori migliori per coordinata e influenza della media dello swarm dipendente dalla dimensione.

Le quattro varianti classiche e CLPSO usano aggiornamenti sincroni a generazioni, inizializzazione uniforme entro i limiti, mappa di vincolo comune con restituzione `0.5`, seed esplicito e contatore FE; SL-PSO inizializza invece lo spostamento a zero come da formulazione. CLPSO usa exemplar per coordinata e refresh del learning set; SL-PSO ordina lo swarm worst-to-best e usa dimostratori migliori e influenza sociale dipendente dalla dimensione. Le sei implementazioni sono baseline di riferimento trasparenti; non sono presentate come una rassegna esaustiva dello stato dell’arte né come implementazioni certificate SPSO-2007/2011.

Il CPSO nativo mantiene la propria derivazione RNG C++ per particella, mentre le baseline usano NumPy. Lo stesso identificatore problema-seme viene associato ai metodi per consentire il pairing delle repliche; non sono garantite coordinate iniziali identiche né common random numbers tra generatori differenti.

### 3.3 Benchmark e protocolli

| Suite | Configurazione smoke | Pre-confirmatory | Confirmatory |
|---|---|---|---|
| COCO BBOB | F1–F3, D2–3, 2 istanze, `200D` | F1–F24, D2/5/10, 5 istanze, `2000D` | F1–F24, D2/3/5/10/20/40, 15 istanze, `10000D` |
| COCO BBOB-LargeScale | F1–F2, D20/40, 1 istanza, `20D` | 8 funzioni selezionate, D20/80/320/640, 3 istanze, `250D` | F1–F24, D20/40/80/160/320/640, 15 istanze, `10000D` |
| CEC 2013 LSGO | F1, 1 seme, 400 FE | F1–F5, 5 semi, 120.000 FE | F1–F15, 25 semi, 3.000.000 FE |

Le tre suite comparano CPSO continuo e sei baseline PSO a tempo discreto in spazio continuo: sette metodi in totale. Ogni campagna impone budget FE divisibile per la popolazione. CEC registra checkpoint a 120.000, 600.000 e 3.000.000 FE nel protocollo confirmatory.

Il preflight BBOB interpreta selettori COCO espliciti, calcola problemi/run/FE totali e rifiuta preventivamente budget non divisibili per la popolazione. I driver creano suite/observer indipendenti per metodo, confrontano il contatore FE del benchmark con quello dell'ottimizzatore e scrivono record JSONL sincronizzati.

I manifest esterni registrano hash della configurazione e dei sorgenti rilevanti, versione e build nativa, versione Python, piattaforma, CPU disponibili, versioni dei pacchetti opzionali e variabili di threading. Il resume è rifiutato se manifest, sorgenti, configurazione, build, dipendenze o ambiente differiscono.

### 3.4 Costi delle campagne complete

Con sette metodi, i piani confirmatory completi richiedono:

- BBOB: 15.120 run e 2,016 miliardi di FE;
- BBOB-LargeScale: 15.120 run e 31,752 miliardi di FE;
- CEC 2013 LSGO: 2.625 run e 7,875 miliardi di FE.

Sono campagne onerose, soprattutto BBOB-LargeScale e CEC. La sequenza raccomandata è smoke con pacchetti ufficiali, subset pre-confirmatory, verifica di storage e tempi, quindi campagne confermative. I numeri di FE rappresentano conti di pianificazione, non lavoro già eseguito.

### 3.5 Manoscritto

Il sorgente LaTeX ora documenta anche i protocolli esterni e le sei baseline PSO a tempo discreto in spazio continuo. La bibliografia include COCO, BBOB-LargeScale, CEC 2013, PSO con inerzia e fattore di costrizione. Il PDF compila e il rendering è stato ispezionato; resta una bozza senza risultati esterni o interni definitivi, discussione conclusiva o evidenza inferenziale.

## 4. Architettura che deve rimanere separata

| Modalità nativa | Comportamento | Relazione col seriale |
|---|---|---|
| `serial` | Generazioni sincrone | Riferimento |
| `master_slave` | Parallelizza valutazioni obiettivo | Traiettoria identica richiesta e testata |
| `particle_parallel` | Parallelizza inizializzazione/propagazione e valutazioni | Traiettoria identica richiesta e testata |
| `asynchronous` | Event-driven | Dinamica differente; multiworker schedule-dependent |
| `islands` | Sottopopolazioni con migrazione | Dinamica differente |

La valutazione esterna richiama solo il riferimento `serial`. Non deve essere usata per concludere che il parallelismo nativo sia più veloce di altri algoritmi; qualità e tempo di callback Python non sono una misura di scaling comparabile.

## 5. Criticità residue

### A. Fedeltà matematica alla formulazione sorgente — bloccante scientifico

Non è stata chiusa la verifica equazione-per-equazione rispetto al testo integrale di Orlando e Ricciardello. Restano da confrontare coefficienti casuali scalari o coordinate-wise, ODE e inerzia, indicizzazione temporale, distribuzioni iniziali, vincoli e istanti di osservazione dell'obiettivo. Il propagatore locale è testato contro RK4 indipendente e proprietà di semigruppo; questi test non dimostrano la corrispondenza completa con l'articolo.

**Input necessario:** PDF integrale o sorgente affidabile dell'articolo. Sino ad allora il manoscritto deve descrivere una ricostruzione operativa documentata, non una replica rigorosa certificata.

### B. Pacchetti ufficiali esterni non ancora verificati

COCO e CEC non sono stati installati nell'ambiente offline. Gli adapter usano le interfacce pubbliche documentate e i test simulati coprono i contratti interni, ma prima delle campagne reali occorre verificare la versione concreta dei pacchetti, i percorsi COCO Observer, i formati di input Cython e i risultati delle funzioni di riferimento. Il pacchetto comunemente usato `cec2013lsgo` è GPL-3.0 e non è vendorizzato; la licenza va considerata nel contesto di distribuzione.

### C. Campagne non eseguite

Nessuna campagna esterna, né la campagna interna definitiva da 9.360 run, è stata completata. I dry-run validano i piani; i test simulati validano il software; nessuno dei due fornisce evidenza di qualità algoritmica.

### D. Baseline e validità comparativa

Le sei baseline sono algoritmi PSO a tempo discreto in spazio continuo: quattro varianti canoniche parametrizzate, CLPSO e SL-PSO. Non implementano integralmente SPSO-2007/2011 né costituiscono da sole una rassegna esaustiva dello stato dell’arte. I metodi condividono popolazione, bounds, seed label e FE; non condividono necessariamente gli stati iniziali o le sequenze casuali perché usano generatori diversi. Questa scelta è documentata. Un eventuale confronto con implementazioni ufficiali SPSO richiederà adapter dedicati e una verifica dei dettagli standardizzati.

Per CEC, la mappa clamp/restitution 0.5 è una scelta del progetto applicata a tutti i metodi. Non è una proprietà del benchmark. Va mantenuta esplicita nelle pubblicazioni e può essere oggetto di un'analisi di sensibilità.

### E. Statistica e ampiezza della base di problemi

L'analisi interna tratta sei funzioni come cluster di livello superiore; la potenza resta limitata. Le campagne esterne ampliano la base di problemi, ma devono ancora essere progettate le tabelle aggregate, i confronti appaiati e il trattamento di obiettivi e istanze COCO secondo il post-processing ufficiale. Per BBOB va conservato il dataset Observer e processato con `cocopp`; non va sostituito da statistiche ad hoc sui soli valori finali.

### F. Piattaforme e licenza del progetto

La CI GitHub Actions Ubuntu/macOS è configurata ma non eseguita sul servizio remoto. Le verifiche locali attuali sono Linux/GCC; macOS sul dispositivo target resta da verificare. La licenza del progetto rimane intenzionalmente non dichiarata fino alla conferma del titolare e all'aggiunta del file legale.

### G. Git e risultati grezzi

L'archivio non include la storia Git. La repository aggiornata deve essere importata in un repository versionato e i risultati grezzi devono essere conservati separatamente, insieme ai manifest, ai COCO observer logs e ai record JSONL.

## 6. Sequenza raccomandata

1. Ottenere il testo integrale dell'articolo sorgente e congelare la formulazione seriale con una matrice di verifica equazione-codice-test.
2. Installare COCO in un ambiente dedicato e verificare `bbob` e `bbob-largescale` smoke su problemi ufficiali; eseguire poi lo smoke CEC solo dopo aver verificato wrapper e licenza.
3. Eseguire le tre campagne pre-confirmatory, controllare qualità, FE, valori noti, stabilità, storage e tempi; definire l'analisi statistica prima delle confermative.
4. Decidere se le baseline PSO devono restare riferimenti algoritmici trasparenti o se è necessario aggiungere implementazioni certificate SPSO.
5. Eseguire le campagne confermative solo dopo il superamento dei gate precedenti e conservare tutti i dati grezzi.
6. Eseguire GitHub CI e build macOS target; confermare la licenza del progetto.
7. Aggiornare il paper con risultati riproducibili, intervalli, dimensioni dell'effetto e limiti; non anticipare alcuna conclusione dai soli smoke test.

## 7. Giudizio

**Implementazione:** la repository ora include il motore CPSO, sei baseline PSO, tre adapter/suite esterni, preflight dei budget, checkpoint e manifest di provenienza. Le verifiche locali automatizzate sono superate.

**Validazione scientifica:** non è ancora chiusa. La fedeltà alla formulazione sorgente, l'esecuzione con le librerie ufficiali e le campagne effettive rimangono prerequisiti prima di formulare conclusioni comparative o di preparare una submission.


## Integrazione benchmark/PSO — versione 0.7.1

- Aggiunta la baseline CLPSO (Liang et al., 2006) con exemplar per coordinata,
  refresh del learning set e snapshot di generazione.
- Aggiunta la baseline SL-PSO (Cheng e Jin, 2015), con ordinamento
  worst-to-best, dimostratori migliori, probabilità di apprendimento e
  influenza sociale dipendenti dalla dimensione. La popolazione resta fissata
  dalla campagna per garantire parità FE; questa scelta differisce dal default
  dimension-dependent dell'articolo ed è registrata nei metadati.
- La suite di confronto comprende ora CPSO continuo e sei varianti PSO a tempo discreto in spazio continuo
  (quattro baseline classiche, CLPSO e SL-PSO). Nessuna è dichiarata come
  universalmente SOTA; SL-PSO è una baseline specifica per problemi scalabili.
- Aggiunte configurazioni di smoke e pilot per CPSO sincrono, asynchronous e islands.
- L'analisi delle campagne tratta l'equivalenza esatta dei metodi sincroni come
  verifica deterministica e non forza un test di Friedman degenere sui pareggi.
