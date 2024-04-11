import cv2
import numpy as np

def compute_optical_flowPyrLK(prev_frame, current_frame):
    # Parametri per il tracciamento dei punti di interesse
    # Lucas-Kanade è l'algoritmo e nei parametri "maxLevel" si riferisce alla profondità della piramide
    lk_params = dict(winSize=(10, 10),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Trovo i punti di interesse nel frame precedente
    # goodFeaturesToTrack() trova i punti di interesse nel prev_frame. 
    # Questi punti di interesse sono selezionati utilizzando l'algoritmo di Shi-Tomasi
    prev_pts = cv2.goodFeaturesToTrack(prev_frame, maxCorners=200, qualityLevel=0.3, minDistance=10, blockSize=8)
    # maxCorners è il numero massimo di punti di interesse
    # qualityLevel è qualità richiesta (tengo bassa)
    # minDistance è minima distanza tra due punti di interesse (ne viene mantenuto solo uno se sono più vicini di minDistance)
    # blockSize è dimensione block per algoritmo. 
    #       Se l'immagine contiene dettagli fini, è consigliabile utilizzare una finestra più piccola per individuare i punti chiave in aree più precise

    # Calcolo il flusso ottico tra i frame
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, prev_pts, None, **lk_params)

    # Filtro i punti di interesse e calcolo il flusso ottico
    good_prev = prev_pts[status == 1]
    good_next = next_pts[status == 1]
    flow = good_next - good_prev
    # calcolato il flusso ottico sottraendo le posizioni dei punti di interesse nel frame
    # corrente dalle posizioni nel frame precedente

    # Calcola la magnitudo e l'angolo del flusso ottico
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Converti l'angolo in gradi
    angle_degrees = np.rad2deg(angle) % 360

    return magnitude, angle_degrees, flow, prev_pts


def overlay_arrows(frame, magnitude, angle_degrees, prev_pts):
    for i, (mag, angle) in enumerate(zip(magnitude, angle_degrees)):
        # Estraggo le coordinate x e y del flusso ottico
        x, y = prev_pts[i].ravel()
        dx, dy = mag * np.cos(np.radians(angle)), mag * np.sin(np.radians(angle))
        # Calcolo il punto finale della freccia
        endpoint = (int(x + dx[0]), int(y + dy[0]))
        # Disegno la freccia
        cv2.arrowedLine(frame, (int(x), int(y)), endpoint, (255, 0, 0), 1)
    return frame





# Percorso delle immagini di input
prev_image_path = "C:/Users/loren/Documents/Data/BlastoData/blasto/D2013.02.19_S0675_I141_1/D2013.02.19_S0675_I141_1_12_0_41324.840721423614.jpg"
current_image_path = "C:/Users/loren/Documents/Data/BlastoData/blasto/D2013.02.19_S0675_I141_1/D2013.02.19_S0675_I141_1_13_0_41324.85114539352.jpg"

# Controllo del caricamento delle immagini
try:
    prev_frame = cv2.imread(prev_image_path, cv2.IMREAD_GRAYSCALE)
    current_frame = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
except FileNotFoundError as e:
    print(f"Errore: File non trovato - {e}")
    exit()

# Calcolo il flusso ottico tra i due frame
magnitude, angle_degrees, flow, prev_pts = compute_optical_flowPyrLK(prev_frame, current_frame)

# Sovrappongo il flusso ottico al secondo frame
overlaid_frame = overlay_arrows(current_frame.copy(), magnitude, angle_degrees, prev_pts)

# Salvataggio del frame con le frecce
output_folder = "C:/Users/loren/Documents/Data/BlastoData/opticalFlowFrames/frame_con_flusso_otticoPyrLK_3.jpg"
cv2.imwrite(output_folder, overlaid_frame)






