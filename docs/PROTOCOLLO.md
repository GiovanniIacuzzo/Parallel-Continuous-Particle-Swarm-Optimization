# Protocollo dei tre studi

## Scopo e invarianti

Il contributo principale è il framework parallelo del CPSO. Il CPSO seriale è il riferimento per verificare lavoro e qualità. Il solver, l'ABI, i sorgenti C++ e il manoscritto non sono modificati. La selezione di una configurazione precede sempre il campione di verifica.

I profili sono dichiarati in `publication_tools/design.py` e copiati integralmente in ciascuna campagna. Smoke, pilot e paper usano intervalli di semi distinti. Le impronte identificano sorgenti, libreria nativa, versioni numeriche e variabili dei thread. I risultati sono scritti atomicamente, con controllo della chiave e checksum. Modifiche incompatibili richiedono una nuova directory. Le classifiche richiedono il disegno completo.

## 1. Sensibilità al passo dt

- Sei funzioni native e due dimensioni; inizializzazione, popolazione e semi appaiati fra tutti i candidati.
- Nel profilo paper: 24 passi da 0,01 a 50, inclusi 0,38125, 1, 3,6 e 30; tre budget indipendenti K=31,127,511 con N=64.
- Ogni budget esegue una nuova run e completa la propria schedule di inerzia. Non si confondono endpoint di budget diversi con checkpoint di un'unica run.
- La metrica primaria è il logaritmo decimale del residuo finale relativo al best iniziale, troncato nell'intervallo [10^-16,1]. Funzioni, dimensioni e budget hanno lo stesso peso. I minimi delle funzioni interne sono zero.
- La metrica secondaria integra lo stesso log-residuo lungo l'asse FE. Profili per funzione e budget rendono visibili le inversioni del vantaggio.
- La scelta minimizza la metrica primaria solo sui semi di selezione. I primi tre candidati e dt=30 vengono congelati per il holdout. La scelta non cambia se un altro candidato vince nel holdout.
- Bootstrap dei semi appaiati con ricampionamento congiunto di tutte le celle; test di Wilcoxon bilaterali e Holm sull'intera famiglia dei contrasti globali, per budget e per cella. Le analisi di selezione sono esplorative; quelle sul candidato congelato usano semi nuovi.

Le inferenze sui semi sono condizionate a questa suite fissa: non rappresentano tutte le possibili funzioni obiettivo. Un p-value non significativo non dimostra equivalenza tra passi.

## 2. Framework parallelo

Si confrontano seriale, master–slave, particle-parallel, asincrono e isole. Nel pilot/paper sono inclusi otto vicinati sociali e worker 1,2,4,8. Le isole hanno quattro sottopopolazioni; vengono variati migrazione assente/intervalli 2/5 nel pilot e 10/50 nel paper e collegamento ring/fully-connected. Il caso senza migrazione non viene duplicato per il nome del grafo di migrazione.

Sono escluse solo combinazioni non supportate dal solver: griglia interna alle isole, più worker che isole, o grado non valido per la dimensione della sottopopolazione. I vicinati sociali e l'architettura di esecuzione rimangono variabili distinte.

### Misure

Per uno stesso problema, seme, ripetizione e vicinato:

- accelerazione rispetto alla seriale: A(p)=T_serial/T_backend,p;
- efficienza rispetto alla seriale: A(p)/p;
- strong scaling sincrono: S(p)=T_backend,1/T_backend,p;
- efficienza interna: S(p)/p.

Si registrano sia il timer del solver sia il tempo wall-clock dell'invocazione Python, che include i costi di allocazione/marshalling. Il criterio operativo di selezione usa quest'ultimo. Il tempo delle verifiche indipendenti, eseguite dopo la run, non entra nel timer.

I due backend sincroni devono riprodurre esattamente best, posizione, storia e ascisse FE della seriale. Una discordanza blocca l'analisi. Per asincrono e isole, il rapporto temporale è denominato runtime ratio: la dinamica può cambiare e il rapporto non è presentato come accelerazione della medesima traiettoria.

Le run vengono riscaldate prima delle misure e l'ordine viene randomizzato. Ripetizioni e sessioni sono aggregate prima entro seme; il bootstrap usa i semi come cluster. Una sessione non viene contata come nuova realizzazione indipendente della qualità. I rapporti sono costruiti sui blocchi appaiati prima della sintesi. Le figure riportano medie geometriche e intervalli bootstrap.

### Selezione e verifica

La selezione considera congiuntamente vicinato, architettura e worker. Prima controlla la qualità media normalizzata e poi minimizza il tempo medio geometrico. La soglia prespecificata ammette al massimo 0,10 unità log10 di perdita rispetto alla migliore qualità di selezione; il limite superiore dell'intervallo bootstrap deve rimanere entro quella soglia. È un criterio di selezione, non un test formale di equivalenza post-selezione.

Si congelano una scelta globale e le scelte per obiettivo/dimensione. Il holdout usa altri semi e altre sessioni, verifica queste sole scelte e i riferimenti seriali, senza riottimizzarle. I contrasti di qualità, i p-value corretti e il superamento del criterio dell'intervallo vengono salvati. Una misura sfavorevole rimane nei risultati.

Se il seriale o un solo worker sono più convenienti, il programma lo riporta. Il nome usato nel confronto è quindi “CPSO selected framework”, senza attribuire parallelismo a una configurazione che non ne usa.

## 3. Confronto PSO

Il confronto nativo trasferisce le scelte congelate del framework. Per un caso già misurato usa la scelta specifica; per nuovi obiettivi/dimensioni usa la scelta globale, senza tuning sul test. Ogni budget della scala è eseguito separatamente.

Le baseline conservate sono Classic gbest, Inertia gbest, Inertia lbest ring, Constriction, CLPSO, FIPS ring, SPSO-2011, SL-PSO, CSO e DMS-PSO. Nessuna baseline viene indebolita in funzione dei risultati. Le implementazioni e i loro adattamenti precedenti restano consultabili in `python/cpso_serial/`.

Il confronto contiene anche il CPSO seriale con lo stesso vicinato della scelta parallela. Questo controllo è escluso dal ranking principale per non assegnare doppio credito al CPSO. Le FE includono l'inizializzazione. Le baseline consumano lo stesso budget esatto anche quando aggiornano solo parte della popolazione.

### Qualità e tempi hanno interpretazioni diverse

Il CPSO esegue gli obiettivi nativi in C++, le baseline usano valutazioni indipendenti Python/NumPy delle stesse formule. I valori finali del CPSO vengono verificati con il valutatore indipendente. Il benchmark di diffusione inversa usa RK4 completo anche nel valutatore NumPy, senza surrogati o precomputazione.

I tempi descrivono le implementazioni consegnate: includono anche linguaggio, callback, vettorizzazione e risorse. Non isolano da soli la superiorità algoritmica. Il confronto a FE fissate misura separatamente la qualità della ricerca.

La figura costo–qualità collega endpoint di run completate a budget diversi. Non rappresenta una traccia anytime a tempo fissato. L'ABI nativa non espone timestamp a ogni generazione e non consente lo stop al target: non si stimano tempi al target interpolando il costo totale. Le percentuali di target nativi sono successi finali al budget specificato.

I ranghi hanno pari peso per funzione. Gli intervalli dei ranghi ricampionano funzioni. I contrasti primari sugli effetti relativi limitati tra [-1,1] aggregano le funzioni entro seme e ricampionano i semi; la loro inferenza è condizionata alla suite interna. Una seconda tabella riporta vittorie/pareggi/sconfitte per funzione e sign test, con correzione Holm. Con sei funzioni questa seconda inferenza ha potenza limitata. Le famiglie includono tutti i budget e le dimensioni del relativo studio, non solo il checkpoint più favorevole.

### BBOB esterno

Il controllo BBOB estende la qualità a 24 funzioni traslate/ruotate. Il profilo paper usa D=5,20, 15 istanze e checkpoint a 64D,192D,512D FE della stessa traiettoria programmata al budget massimo. Le istanze sono distinte da quelle del pilot. Il vicinato e dt vengono dalla scelta globale; la callback è obbligatoriamente seriale con l'ABI corrente. Le prestazioni di asincrono o isole non vengono attribuite a questo controllo seriale.

Il logger COCO ufficiale fornisce il minimo, letto dopo la chiusura della run senza consumare FE aggiuntive. ECDF su 21 target da 10^2 a 10^-8; gli insuccessi restano nel denominatore. ERT in FE e tempo viene calcolato per funzione, dimensione e target: costo dei successi più budget/costo finale degli insuccessi, diviso per il numero di successi. Con zero successi è indefinito. Non si presenta un ERT unico su funzioni eterogenee.

Le classifiche usano undici metodi, istanze aggregate entro funzione, bootstrap delle funzioni e contrasti per funzione con Holm. Il controllo BBOB è un risultato esterno sulla qualità del CPSO seriale di riferimento; non colma il confronto multicore esterno, che richiederebbe una futura estensione dell'interfaccia C++.

## Figure e conservazione

Tutti i grafici generati sono PDF vettoriali con caratteri incorporati e testo inglese. Le didascalie da scrivere nel manoscritto devono indicare: suite, dimensione, budget, campione di selezione/holdout, unità di bootstrap, significato del riferimento e natura del timer. Le statistiche tabellari rimangono CSV per l'inserimento nativo in LaTeX.

Il pacchetto non contiene nuovi risultati scientifici: le run smoke verificano il funzionamento su una macchina diversa dal Mac dell'autore. I risultati M2 devono essere eseguiti e controllati prima di formulare conclusioni sul metodo vincente.
