import numpy as np
import cv2

def compute_optical_flow(prev_frame, current_frame, window_size, search_size, overlap):
    """
    Calcola il campo di velocità ottico tra due frame consecutivi.

    Parametri:
        prev_frame: Il frame precedente.
        current_frame: Il frame corrente.
        window_size: Dimensione della finestra per la ricerca (es: (15, 15)).
        search_size: Dimensione della regione di ricerca (es: (30, 30)).
        overlap: Sovrapposizione tra le finestre (0-1).

    Restituisce:
        - La magnitudo del flusso ottico.
        - L'angolo del flusso ottico in gradi.
    """
    # Inizializza le liste per la magnitudo e l'angolo del flusso ottico
    magnitudes = []
    angles = []

    # Calcola la dimensione dell'overlap
    overlap_pixels = (int(window_size[0] * overlap), int(window_size[1] * overlap))

    # Itera attraverso i frame con finestre sovrapposte
    for y in range(0, current_frame.shape[0] - search_size[0], overlap_pixels[0]):
        for x in range(0, current_frame.shape[1] - search_size[1], overlap_pixels[1]):
            # Definisci le regioni di interesse nei frame
            window_prev = prev_frame[y+window_size[0]//2:y+window_size[0]*3//2, x+window_size[1]//2:x+window_size[1]*3//2]
            window_current = current_frame[y:y+search_size[0], x:x+search_size[1]]

            #window_prev = prev_frame[y:y+window_size[0], x:x+window_size[1]]
            #window_current = current_frame[y:y+search_size[0], x:x+search_size[1]]

            # Calcola la cross-correlazione tra le finestre
            correlation = cv2.matchTemplate(window_current, window_prev, cv2.TM_SQDIFF_NORMED)

            # Trova la posizione del picco di correlazione
            _, _, _, peak_loc = cv2.minMaxLoc(correlation)
            peak_y, peak_x = peak_loc

            # Calcola il vettore di spostamento
            displacement = (peak_x - window_size[0] // 2, peak_y - window_size[1] // 2)

            # Calcola magnitudo e angolo del flusso ottico
            magnitude = np.sqrt(displacement[0] ** 2 + displacement[1] ** 2)
            angle = np.arctan2(displacement[1], displacement[0]) * 180 / np.pi

            # Aggiungi i risultati alle liste
            magnitudes.append(magnitude)
            angles.append(angle)

    return magnitudes, angles


def overlay_arrows(frame, magnitudes, angles, step=16):
    """
    Sovrapponi le frecce che rappresentano il flusso ottico sul frame.

    Parametri:
        frame: Il frame su cui sovrapporre le frecce.
        magnitudes: Lista delle magnitudini del flusso ottico.
        angles: Lista degli angoli del flusso ottico.
        step: Passo per campionare il frame per posizionare le frecce.

    Restituisce:
        Il frame con le frecce del flusso ottico sovrapposte.
    """
    # Copia del frame per evitare la modifica dell'originale
    overlay_frame = frame.copy()

    # Sovrapponi le frecce sul frame
    for y in range(0, overlay_frame.shape[0], step):
        for x in range(0, overlay_frame.shape[1], step):
            # Estrai la magnitudo e l'angolo del flusso ottico per questa posizione
            magnitude = magnitudes[y // step * (overlay_frame.shape[1] // step) + x // step]
            angle = angles[y // step * (overlay_frame.shape[1] // step) + x // step]

            # Calcola il punto finale dell'arco
            endpoint = (int(x + magnitude * np.cos(np.radians(angle))),
                        int(y + magnitude * np.sin(np.radians(angle))))

            # Disegna la freccia
            cv2.arrowedLine(overlay_frame, (x, y), endpoint, (0, 255, 0), 1)

    return overlay_frame


# Percorsi delle immagini
prev_image_path = "C:/Users/loren/Documents/Data/BlastoData/blasto/D2013.02.19_S0675_I141_1/D2013.02.19_S0675_I141_1_12_0_41324.840721423614.jpg"
current_image_path = "C:/Users/loren/Documents/Data/BlastoData/blasto/D2013.02.19_S0675_I141_1/D2013.02.19_S0675_I141_1_13_0_41324.85114539352.jpg"

# Carica i frame
prev_frame = cv2.imread(prev_image_path, cv2.IMREAD_GRAYSCALE)
current_frame = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)

# Parametri per il calcolo del flusso ottico
window_size = (16, 16)
search_size = (2 * window_size[0], 2 * window_size[1])
overlap = 0.5

# Calcola il flusso ottico
magnitudes, angles = compute_optical_flow(prev_frame, current_frame, window_size, search_size, overlap)

# Sovrapponi le frecce sul frame corrente
overlayed_frame = overlay_arrows(current_frame, magnitudes, angles)

# Salvo il frame con le frecce 
# output_path = "frame_con_flusso_ottico.jpg"
output_folder = "C:/Users/loren/Documents/Data/BlastoData/opticalFlowFrames/frame_con_flusso_ottico_Manual.jpg"
cv2.imwrite(output_folder, overlayed_frame)

