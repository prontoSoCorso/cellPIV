import os
import pandas as pd
import shutil

import sys
# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_00_preprocessing as conf

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

if __name__ == "__main__":
    # Percorso della directory sorgente (con i video equatoriali)
    src_dir = "/home/phd2/Scrivania/CorsoData/ScopeData_equator"
    
    # Percorso della directory di destinazione
    dest_dir = "/home/phd2/Scrivania/CorsoData/blastocisti"
    
    # Percorso del file Excel
    excel_path = conf.path_original_excel

    # Organizza i video
    organize_videos_by_classification(src_dir, dest_dir, excel_path)



'''
Video non spostati e motivazioni:
D2019.03.13_S02233_I0141_D_5 non trovato nella directory sorgente
D2020.11.03_S02668_I0141_D_1 non trovato nella directory sorgente
D2020.11.03_S02668_I0141_D_2 non trovato nella directory sorgente
D2020.11.03_S02668_I0141_D_3 non trovato nella directory sorgente

Totale video copiati: 6048
'''


