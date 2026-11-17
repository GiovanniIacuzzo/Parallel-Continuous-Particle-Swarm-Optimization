# Modifiche consegnate

Base: repository GitHub `GiovanniIacuzzo/Parallel-Continuous-Particle-Swarm-Optimization`, commit `5c09645b2dacb9fc9399e780d179b09be06d8f50`.

## Percorso attivo

- Nuovo punto di ingresso `publication.py`; `study.py` rimane come alias.
- Moduli in `publication_tools/` separati per protocolli, sensibilità, architetture, confronti, logger e figure.
- Dependency analysis aggiornata per includere pandas, già necessario alle analisi recenti.
- Nuova guida M2, protocollo scientifico esplicito, manifest di pulizia e rapporto di verifica.

## Rafforzamento dei tre nuclei

1. dt: griglia estesa, budget indipendenti, selezione congelata, holdout separato, metriche finali/anytime, bootstrap appaiato e Holm, profili per funzione.
2. Parallelismo: tutte le architetture applicabili; otto vicinati nel pilot/paper; variazione di worker, migrazione e sessioni; confronto esatto sincrono; tempi nativi e wall-clock; scelta globale e per caso; holdout senza tuning.
3. PSO: trasferimento della configurazione congelata al confronto nativo; dieci baseline conservate; controllo seriale escluso dal ranking principale; ranghi e incertezza, statistiche per seme e funzione, target e confronti di costo/qualità. BBOB seriale esterno con 24 funzioni, logger ufficiale, ECDF ed ERT per funzione.

I PDF sono vettoriali, con etichette inglesi, legende separate dagli assi e palette coerente. Le tabelle rimangono dati CSV.

## Protezioni e pulizia

`paper/`, tutti i file C++ e CMake sono invariati. Gli hash dei 61 file protetti permettono un controllo automatico. I runner delle revisioni precedenti e gli atlanti sovrapposti sono rimossi dal percorso attivo; una copia è in `recovery/CPSO_legacy_recovery.zip` nel progetto estratto. I controlli nativi e del logger che restano utili sono conservati.

La consegna è una copia modificata della repository, non un aggiornamento remoto su GitHub. Per ottenere anche le rimozioni va estratta in una directory nuova.

## Limiti che rimangono espliciti

L'ABI nativa permette callback personalizzate solo seriali. Non è stata aggirata questa limitazione e non sono state attribuite al parallelo le misure BBOB/CEC seriali. I confronti temporali C++/NumPy descrivono le implementazioni. I PDF nativi costo–qualità mostrano endpoint di run completate, senza inventare timestamp intermedi o interpolare tempi al target.

Il profilo paper è una proposta prespecificata più ampia: richiede l'esecuzione sul Mac e una valutazione dei risultati. Le prove smoke consegnate nel rapporto sono verifiche software su Linux, non evidenze di superiorità del CPSO e non risultati da trasferire nel manoscritto.
