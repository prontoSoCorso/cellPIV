import os
import pandas as pd
import shutil
from PIL import Image, ImageFile


# ========== Organizzazione dei video di blasto e no_blasto in cartelle distinte ==========

def organize_videos_by_classification(src_dir, dest_dir, excel_path):
    """
    Organizza i video in due cartelle ("blasto" e "no_blasto") basandosi sulla colonna "blasto ny" nel file Excel.
    """
    try:
        # Carica il file Excel
        data = pd.read_excel(excel_path)

        # Filtra solo le colonne di interesse
        relevant_data = data[["slide_well", "blasto ny"]]

        # Directory di destinazione
        blasto_dir = os.path.join(dest_dir, "blasto")
        no_blasto_dir = os.path.join(dest_dir, "no_blasto")
        os.makedirs(blasto_dir, exist_ok=True)
        os.makedirs(no_blasto_dir, exist_ok=True)

        # Lista dei video non spostati
        not_moved_videos = []
        moved_count = 0

        # Itera su ogni riga del file Excel
        for _, row in relevant_data.iterrows():
            video_name = row["slide_well"]
            classification = row["blasto ny"]

            # Determina la cartella di destinazione in base alla classificazione
            if classification == 1:
                target_dir = blasto_dir
            elif classification == 0:
                target_dir = no_blasto_dir
            else:
                continue  # Salta righe senza classificazione valida

            # Estrai l'anno dal nome del video
            try:
                year = video_name.split("_")[0][1:5]
                video_path = os.path.join(src_dir, year, video_name)

                # Controlla se la cartella esiste nella directory sorgente
                if os.path.isdir(video_path):
                    # Sposta la cartella nella directory target
                    dest_path = os.path.join(target_dir, os.path.basename(video_path))
                    shutil.move(video_path, dest_path)
                    print(f"Video spostato: {video_name} -> {dest_path}")
                    moved_count += 1
                else:
                    reason = f"{video_name} non trovato nella directory sorgente"
                    print(f"Attenzione: {reason}")
                    not_moved_videos.append(reason)
            except Exception as e:
                reason = f"Errore nell'elaborazione di {video_name}: {e}"
                print(f"Attenzione: {reason}")
                not_moved_videos.append(reason)

        # Salva i video non spostati in un file di log
        log_path = os.path.join(dest_dir, "videos_not_moved.txt")
        with open(log_path, "w") as log_file:
            log_file.write("Video non spostati e motivazioni:\n")
            log_file.write("\n".join(not_moved_videos))
            log_file.write(f"\n\nTotale video copiati: {moved_count}")

        print(f"Organizzazione completata. I video sono stati separati in '{blasto_dir}' e '{no_blasto_dir}'.")
        print(f"Dettagli sui video non spostati salvati in: {log_path}")

    except Exception as e:
        print(f"Errore durante l'organizzazione: {e}")




'''
Video non spostati e motivazioni:
D2019.03.13_S02233_I0141_D_5 non trovato nella directory sorgente
D2020.11.03_S02668_I0141_D_1 non trovato nella directory sorgente
D2020.11.03_S02668_I0141_D_2 non trovato nella directory sorgente
D2020.11.03_S02668_I0141_D_3 non trovato nella directory sorgente

Totale video copiati: 6048
'''




# ========== Check di potenziali immagini corrotte ==========

def check_for_corrupted_images(root_dir, log_file_path):
    """
    Verifica se ci sono immagini corrotte nelle cartelle all'interno di `root_dir`.
    Salva un log con i risultati.
    
    Parameters:
        root_dir (str): Directory principale con le cartelle da verificare.
        log_file_path (str): Percorso del file di log.
    """
    corrupted_folders = []
    total_images = 0
    corrupted_images = 0

    # Apri il file di log
    with open(log_file_path, "w") as log_file:
        log_file.write("Verifica immagini corrotte:\n\n")
        
        # Scansiona tutte le cartelle e sottocartelle
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                folder_corrupted = False
                log_file.write(f"Verifica cartella: {folder_name}\n")
                
                # Verifica tutte le immagini nella cartella
                for root, _, files in os.walk(folder_path):
                    for file_name in files:
                        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                            total_images += 1
                            image_path = os.path.join(root, file_name)
                            try:
                                with Image.open(image_path) as img:
                                    img.verify()  # Verifica se l'immagine è leggibile
                            except Exception as e:
                                log_file.write(f"  Immagine corrotta: {image_path} - Errore: {e}\n")
                                corrupted_images += 1
                                folder_corrupted = True
                
                # Se la cartella contiene immagini corrotte, aggiungila alla lista
                if folder_corrupted:
                    corrupted_folders.append(folder_name)
                    log_file.write(f"  La cartella '{folder_name}' contiene immagini corrotte.\n")
                else:
                    log_file.write(f"  La cartella '{folder_name}' non contiene immagini corrotte.\n")
                
                log_file.write("\n")
        
        # Riassunto
        log_file.write("\n--- Riassunto ---\n")
        log_file.write(f"Totale immagini controllate: {total_images}\n")
        log_file.write(f"Totale immagini corrotte: {corrupted_images}\n")
        log_file.write(f"Cartelle con immagini corrotte: {len(corrupted_folders)}\n")
        log_file.write(f"Cartelle corrotte: {', '.join(corrupted_folders) if corrupted_folders else 'Nessuna'}\n")

    print(f"Verifica completata. Log salvato in: {log_file_path}")



'''
Verifica immagini corrotte:

Verifica cartella: no_blasto
  La cartella 'no_blasto' non contiene immagini corrotte.

Verifica cartella: blasto
  La cartella 'blasto' non contiene immagini corrotte.


--- Riassunto ---
Totale immagini controllate: 3805031
Totale immagini corrotte: 0
Cartelle con immagini corrotte: 0
Cartelle corrotte: Nessuna
'''



# ========== Fix delle immagini ==========


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


