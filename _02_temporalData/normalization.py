










'''
Normalization Strategies:
    Normalizzazione su tutto il dataset
        Vantaggi:
            Comparabilità globale: Facilita l'uso di modelli di machine learning, che spesso beneficiano di feature scalate uniformemente.
            Riduzione della varianza: Aiuta a stabilizzare i modelli predittivi, riducendo l'influenza degli outlier.

        Svantaggi:
            Diluzione delle differenze specifiche: Potrebbe nascondere variazioni importanti tra i wells di diversi pazienti.
            Rischio di overfitting: Se le caratteristiche uniche di un paziente sono rilevanti, questo approccio potrebbe non catturarle bene.

            

    Normalizzazione per singolo well
        Vantaggi:
            Preserva la variabilità intra-well: Mantiene le specificità di ogni well, che potrebbero essere cruciali per la predizione accurata.
            Riduzione della variabilità locale: Facilita l'analisi delle variazioni all'interno di ogni well, potenzialmente importanti per la formazione delle blastocisti.

        Svantaggi:
            Scarsa comparabilità tra wells: Potrebbe rendere difficile identificare pattern globali, necessitando di metodi di aggregazione più complessi.
            Rischio di sovradimensionamento: Troppa enfasi sulle caratteristiche individuali di un well potrebbe portare a modelli troppo specifici e meno generalizzabili.


    Normalizzazione per paziente
        Vantaggi:
            Equilibrio tra variabilità intra e inter-paziente: Mantiene le differenze significative tra pazienti, pur riducendo la varianza interna.
            Facilitazione del modello: Aiuta i modelli di machine learning a captare pattern rilevanti che sono consistenti tra i wells di un singolo paziente.

        Svantaggi:
            Variabilità interna meno evidente: Potrebbe mascherare variazioni all'interno dei wells di un paziente.
            Richiede dati consistenti: Questo approccio funziona meglio con un numero significativo di wells per paziente, altrimenti la normalizzazione potrebbe non essere robusta.



Decisione finale: NORMALIZZAZIONE PER PAZIENTE
    - Controlla la variabilità inter-individuale: Riduce l'impatto delle differenze tra pazienti, 
        facilitando l'identificazione di pattern rilevanti per la predizione.
    - Mantiene l'informazione intra-paziente: Conserva le specificità dei wells, che possono essere 
        cruciali per la formazione delle blastocisti.
    - Facilita l'apprendimento del modello: Modelli di machine learning spesso beneficiano di dati 
        normalizzati a livello di gruppo, migliorando la stabilità e la generalizzabilità.
'''































