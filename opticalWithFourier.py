import numpy as np
import cv2

def fourier_shift(gray_prev_frame, gray_current_frame, overlap):

    """
    Calcola la cross-correlazione tra due immagini usando lo shift di Fourier.

    Parametri:
        gray_prev_frame: La prima immagine in scala di grigi.
        gray_current_frame: La seconda immagine in scala di grigi.

    Restituisce:
        La distribuzione di cross-correlazione.
    """

    # Calcolare la FFT di entrambe le immagini
    fft_prev_frame = np.fft.fft2(gray_prev_frame)
    fft_current_frame = np.fft.fft2(gray_current_frame)

    # Calcolare lo shift di Fourier
    shifted_fft = np.fft.fftshift(fft_current_frame) * np.fft.fftshift(np.conj(fft_prev_frame))

    # Applicare l'overlap
    if overlap > 0 and overlap < 1:
        # Calcolare la dimensione della cross-correlazione
        cross_correlation_size = int(np.max(gray_prev_frame.shape) / overlap)
        # Eseguire l'overlap
        shifted_fft = shifted_fft[:cross_correlation_size, :cross_correlation_size]

    # Calcolare la cross-correlazione
    cross_correlation_distribution = np.fft.ifft2(shifted_fft)

    return cross_correlation_distribution



def compute_optical_flow_matpiv(prev_frame, current_frame, overlap):
    """
    Calcola il flusso ottico tra due immagini usando MatPIV, con implementazione della FFT per la correlazione.

    Parametri:
        prev_frame: La prima immagine.
        current_frame: La seconda immagine.

    Restituisce:
        - La magnitudo del flusso ottico.
        - L'angolo del flusso ottico in gradi.
        - Il campo vettoriale del flusso ottico.
    """

    # Definire i parametri di MatPIV
    interrogation_region_size = 15
    search_region_size = interrogation_region_size * 2

    # Calcolare la cross-correlazione
    cross_correlation_distributions = fourier_shift(current_frame, -np.fft.fftshift(prev_frame), overlap)

    # Trovare i picchi delle distribuzioni di cross-correlazione
    displacements = np.argmax(cross_correlation_distributions, axis=2)

    # Calcolare il campo vettoriale della velocità
    velocity_field = displacements / search_region_size

    # Calcolare la magnitudo e l'angolo del flusso ottico
    magnitude, angle = cv2.cartToPolar(velocity_field[..., 0], velocity_field[..., 1])

    # Convertire l'angolo in gradi
    angle_degrees = np.rad2deg(angle) % 360

    return magnitude, angle_degrees, velocity_field



# Definire la funzione per la sovrapposizione delle frecce
def overlay_arrows(frame, magnitude, angle_degrees, step=16):
    # Overlay arrows 
    for y in range(0, frame.shape[0], step):
        for x in range(0, frame.shape[1], step):
            # Get the flow at this point
            flow_magnitude = magnitude[y, x]
            flow_angle = angle_degrees[y, x]

            # Calculate the endpoint of the arrow
            endpoint = (int(x + flow_magnitude * np.cos(np.radians(flow_angle))),
            int(y + flow_magnitude * np.sin(np.radians(flow_angle))))

            # Draw the arrow
            cv2.arrowedLine(frame, (x, y), endpoint, (255, 0, 0), 1)

    return frame




# Caricare le immagini
prev_image_path = "C:/Users/loren/Documents/Data/BlastoData/blasto/D2013.02.19_S0675_I141_1/D2013.02.19_S0675_I141_1_1_0_41324.7260405787.png"
current_image_path = "C:/Users/loren/Documents/Data/BlastoData/blasto/D2013.02.19_S0675_I141_1/D2013.02.19_S0675_I141_1_16_0_41324.88240604167.png"

prev_frame = cv2.imread(prev_image_path, cv2.IMREAD_GRAYSCALE)
current_frame = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)

# Calcolare il flusso ottico
overlap = 0.5
magnitude, angle_degrees, velocity_field = compute_optical_flow_matpiv(prev_frame, current_frame, overlap)

# Visualizzare la seconda immagine con sovrapposto il flusso ottico
overlaid_frame = overlay_arrows(current_frame.copy(), magnitude, angle_degrees)

cv2.imshow("Flusso ottico con overlay di frecce", overlaid_frame)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
