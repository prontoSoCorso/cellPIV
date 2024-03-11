import tensorflow as tf

def compute_optical_flow_matpiv_tensorflow(prev_frame, current_frame):

    # Definire i parametri di MatPIV
    interrogation_region_size = 15
    search_region_size = interrogation_region_size * 2

    # Convertire le immagini in tensori
    prev_frame_tensor = tf.convert_to_tensor(prev_frame)
    current_frame_tensor = tf.convert_to_tensor(current_frame)

    # Calcolare la cross-correlazione usando TensorFlow
    cross_correlation_distributions = tf.signal.fft2d(current_frame_tensor) * tf.signal.fft2d(tf.math.conj(prev_frame_tensor))

    # Applicare l'overlap del 50%
    cross_correlation_distributions = cross_correlation_distributions[::2, ::2]

    # Trovare i picchi delle distribuzioni di cross-correlazione
    displacements = tf.argmax(cross_correlation_distributions, axis=2)

    # Calcolare il campo vettoriale della velocità
    velocity_field = displacements / search_region_size

    # Calcolare la magnitudo e l'angolo del flusso ottico
    magnitude, angle = tf.cartesian_to_polar(velocity_field[..., 0], velocity_field[..., 1])

    # Convertire l'angolo in gradi
    angle_degrees = tf.math.mod(tf.math.degrees(angle), 360)

    return magnitude, angle_degrees, velocity_field