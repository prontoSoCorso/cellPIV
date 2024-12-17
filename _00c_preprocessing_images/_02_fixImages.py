import os
from PIL import Image, ImageFile

# Funzione per correggere immagini troncate
def fix_truncated_jpeg(file_path):
    """
    Tenta di correggere un file JPEG troncato.
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(file_path) as img:
            # Salva il file correggendo eventuali problemi di fine file
            img.save(file_path, "JPEG")
        return file_path
    except Exception as e:
        print(f"Errore nella correzione del file {file_path}: {e}")
        return None

# Funzione per iterare su tutte le cartelle e correggere le immagini
def fix_images_in_directory(root_path):
    """
    Itera su tutte le cartelle sotto il path root_path e corregge le immagini troncate.
    """
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(".jpg"):  # Filtra solo i file .jpg
                file_path = os.path.join(root, file)
                print(f"Correggendo immagine: {file_path}")
                fixed_image_path = fix_truncated_jpeg(file_path)
                if fixed_image_path:
                    print(f"Immagine corretta salvata come: {fixed_image_path}")
                else:
                    print(f"Immagine non corretta: {file_path}")

# Path della cartella root
image_path = "/home/phd2/Scrivania/CorsoData/blastocisti"

# Esegui la correzione su tutte le immagini nelle cartelle
fix_images_in_directory(image_path)
