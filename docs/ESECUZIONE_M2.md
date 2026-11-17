# Esecuzione sul MacBook Air M2

Usare una nuova directory estratta dallo ZIP, senza sovrapporla al vecchio checkout. Conservare separatamente i risultati precedenti. La nuova campagna non li importa e non li cancella. Tutti i comandi vanno eseguiti nella directory del nuovo README. Non sono necessari push o modifiche a GitHub.

## 1. Preparare e verificare l'ambiente

Eseguire l'installazione del README. Controllare `python -c 'import platform; print(platform.machine())'`: deve risultare `arm64`. La libreria si compila sul Mac; lo ZIP non contiene binari Linux. Usare alimentazione di rete, modalità di risparmio energetico disattivata e nessun'altra simulazione contemporanea. Tenere le condizioni il più possibile costanti tra sessioni. Il programma registra informazioni sull'alimentazione e sui core disponibili, ma non misura in modo affidabile la temperatura del chip.

```bash
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
python publication.py check
```

Il controllo verifica caricamento del binario, una run nativa e gli hash di `paper/` e dei sorgenti C++. Non modificare queste cartelle durante la campagna.

## 2. Test breve dell'intero percorso

```bash
python publication.py dt --profile smoke
python publication.py parallel --profile smoke
python publication.py compare --profile smoke
python publication.py compare --profile smoke --suite inverse
```

Se COCO è installato:

```bash
python publication.py bbob --profile smoke
```

Il test ridotto usa pochissimi semi e budget: verifica software e output, non produce risultati scientifici da citare. Non trasferisce la propria scelta di dt al pilot.

## 3. Prima campagna da inviare per il controllo

```bash
python publication.py plan --profile pilot
python publication.py dt --profile pilot
python publication.py parallel --profile pilot
python publication.py compare --profile pilot
python publication.py bbob --profile pilot
python publication.py pack --profile pilot
```

Procedere nell'ordine indicato. Il programma mostra l'avanzamento; non sono promessi tempi senza aver misurato questo ambiente. Il pilot contiene 3.456 run di selezione dt, fino a 1.728 di holdout, 6.688 misure nella selezione delle architetture, una validazione più piccola sulle scelte congelate, 5.184 run del confronto nativo e 1.584 run BBOB opzionali. Le run hanno costi molto diversi.

**È possibile fermarsi dopo dt oppure dopo il confronto parallelo e creare subito lo ZIP.** Questo permette di esaminare il costo e la qualità prima di lanciare i confronti più lunghi. `pack` conserva anche risultati parziali.

Il file da inviare è `risultati_CPSO_pilot.zip` nella directory principale. Comprende dati grezzi, protocolli, verifiche statistiche, metadati, logger ufficiali presenti e figure. Non serve selezionare manualmente i file.

Il confronto sulla diffusione inversa è separato perché la valutazione NumPy del sistema RK4 è costosa:

```bash
python publication.py compare --profile pilot --suite inverse
python publication.py pack --profile pilot
```

## 4. Ripresa e suddivisione

Interrompere con Ctrl+C e rilanciare lo stesso comando: i risultati completati vengono riutilizzati. Ogni record è scritto atomicamente. Le interruzioni del logger COCO vengono archiviate prima di ripetere esclusivamente la run non completata.

È possibile distribuire le funzioni native su lanci successivi, eseguiti sempre in sequenza:

```bash
python publication.py compare --profile pilot --benchmark sphere
python publication.py compare --profile pilot --benchmark rastrigin
```

Completare analogamente `rosenbrock`, `ackley`, `griewank`, `salomon`, oppure rilanciare senza `--benchmark` per completare quelle mancanti. La statistica complessiva viene generata soltanto quando tutti i casi del protocollo sono completi.

BBOB può essere suddiviso per funzione:

```bash
python publication.py bbob --profile pilot --function 1
```

Rilanciare senza `--function` completa le altre funzioni. Anche in questo caso nessuna classifica complessiva viene prodotta su un campione incompleto.

Non ricompilare, aggiornare dipendenze o modificare gli script durante una campagna: la ripresa rifiuta le impronte incompatibili. Per un nuovo protocollo usare un nuovo percorso dentro `results/`, mantenendo lo stesso `--output` in tutte le fasi:

```bash
python publication.py dt --profile pilot --output results/publication_v2/pilot_seconda_sessione
```

## 5. Campagna estesa dopo il controllo del pilot

```bash
python publication.py plan --profile paper
python publication.py dt --profile paper
python publication.py parallel --profile paper --phase selection --session 1
```

La selezione parallela del profilo paper richiede due sessioni. Eseguire la seconda in condizioni comparabili, quindi la sessione indipendente di verifica:

```bash
python publication.py parallel --profile paper --phase selection --session 2
python publication.py parallel --profile paper --phase holdout --session 3
python publication.py compare --profile paper
python publication.py bbob --profile paper
python publication.py pack --profile paper
```

La selezione dt estesa comprende 24 candidati, 3 budget, 6 funzioni, 2 dimensioni e 30 semi: 25.920 run. Il profilo è più oneroso del pilot e va pianificato dopo la prima misura. Il confronto BBOB esteso usa 15 istanze per funzione e dimensione. Le decisioni sono congelate prima del test; risultati sfavorevoli non innescano un tuning automatico sul campione di verifica.

## 6. Rigenerare le figure

```bash
python publication.py figures --profile pilot
```

I PDF vengono salvati in `results/publication_v2/pilot/figures/`, suddivisi secondo i tre nuclei dell'articolo. I CSV sono dati per tabelle LaTeX e controlli; non vengono convertiti in immagini. `paper/` non riceve output automatici.

## CEC e runner precedenti

I runner in `experiments/05_external_benchmarks/` rimangono disponibili come controlli esterni aggiuntivi. Per CEC:

```bash
python -m pip install 'setuptools>=65,<81' 'Cython>=3,<4'
python -m pip install --no-build-isolation -r requirements-benchmarks-cec2013.txt
python experiments/05_external_benchmarks/run_cec2013_lsgo.py --help
```

Le configurazioni storiche non vengono automaticamente allineate alla nuova scelta congelata. Il CEC completo ha un costo elevato; non è necessario per controllare il primo pilot. Le callback CEC, come quelle BBOB, restano seriali con l'ABI corrente.
