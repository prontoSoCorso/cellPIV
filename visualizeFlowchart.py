import graphviz


# FLOWCHART 1

# Crea un diagramma di flusso del processo usando Graphviz
flowchart = graphviz.Digraph(format='png', graph_attr={'dpi': '600'})

# Step 1: Seleziona valori positivi
flowchart.node('C', 'Calcola min_val, max_val \nignorando gli zeri')
flowchart.node('D', 'min_val = np.min(valori_positivi)\nmax_val = np.max(valori_positivi)')

# Step 3: Verifica se il massimo è nelle ultime 10 colonne
flowchart.node('E', 'Il max_val è nelle ultime 10 colonne?')
flowchart.node('F', 'Sì: Imposta le ultime 10 colonne a zero \nIncrementa check_number_last_column_changed')
flowchart.node('G', 'No')

# Step 4: Imposta le ultime 10 colonne a zero
flowchart.node('H', 'Imposta le ultime 10 colonne a zero')

# Step 5: Ricalcola min e max se il massimo era nelle ultime 10 colonne
flowchart.node('I', 'Ricalcola min_val, max_val ignorando gli zeri')

# Step 6: Applica la normalizzazione
flowchart.node('J', 'Applica la normalizzazione Min-Max')
flowchart.node('K', 'normalizzato = (gruppo - min_val) / (max_val - min_val)')

# Step 7: Mantieni invariati gli zeri
flowchart.node('L', 'Mantieni invariati gli zeri nell\'array normalizzato')

# Step 8: Ritorna il gruppo normalizzato
flowchart.node('M', 'Ritorna il gruppo normalizzato')

# Connessioni
flowchart.edges([('C', 'D'), ('D', 'E'), ('E', 'F'), ('E', 'G'),
                ('F', 'H'), ('H', 'I'), ('I', 'J'), ('G', 'J'), ('J', 'K'), ('K', 'L'), ('L', 'M')])

# Salva il diagramma di flusso
flowchart.render('processo_diagramma_flusso', view=True)



'''
# FLOWCHART 2

# Crea un diagramma di flusso generale del processo di selezione e normalizzazione dei dati
flowchart = graphviz.Digraph(format='png', graph_attr={'dpi': '600'})

# Step 1: Selezione dei dati
flowchart.node('A', 'Selezione dei Dati\n(Seleziona video degli embrioni)')

# Step 2: Filtraggio
flowchart.node('B', 'Filtraggio\n(Applica regole: \nframe > 300, dimensione = 500x500,\nrimuovi i primi 5 frame)')

# Step 3: Calcolo del Flusso Ottico
flowchart.node('C', 'Calcolo del Flusso Ottico\n(Calcola il flusso usando Lucas-Kanade)')

# Step 4: Estrazione delle Caratteristiche
flowchart.node('D', 'Estrazione delle Caratteristiche\n(Magnitudine Media, Vorticità, Ibrida, Somma Magnitudine Media)')

# Step 5: Normalizzazione a livello di paziente
flowchart.node('E', 'Normalizzazione\n(Min-Max per paziente,\nverifica spostamenti nelle ultime 10 colonne)')

# Step 6: Dataset Finale per ML
flowchart.node('F', 'Dataset Finale\n(Pronto per l\'analisi con modelli di machine learning)')

# Connessioni tra i vari step
flowchart.edges([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F')])

# Salva il diagramma di flusso
flowchart.render('processo_generale_diagramma_flusso', view=True)

'''

'''
# FLOWCHART 3

# Creazione di un nuovo diagramma
flowchart = graphviz.Digraph(format='png', graph_attr={'dpi': '600'})

# Estrazione dei file
flowchart.node('A', 'Estrazione video dai file PDB')

# Miglioramento del contrasto
flowchart.node('B', 'Ritaglio dello sfondo')
flowchart.node('C', 'Applicazione CLAHE\n(Miglioramento del contrasto)')
flowchart.node('D', 'Binarizzazione e rimozione artefatti\n(Trasformazioni morfologiche)')
flowchart.node('E', 'Identificazione contorni\n(Blob più grandi)')
flowchart.node('F', 'Applicazione della maschera')

# Collegamenti tra i passaggi
flowchart.edges([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F')])

# Nodo finale
flowchart.node('G', 'Immagini pronte per analisi')
flowchart.edge('F', 'G')  # Collegamento all'output finale

# Rendering e visualizzazione
flowchart.render('flowchart immagini', view=True)
'''