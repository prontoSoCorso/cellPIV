import os
import shutil

def copy_equatorial_frames(src_dir, dest_dir):
    """Copia solo i frame equatoriali (_0_) mantenendo la struttura delle directory."""
    try:
        # Itera attraverso tutti gli anni nella directory sorgente
        for year in os.listdir(src_dir):
            year_path = os.path.join(src_dir, year)
            if os.path.isdir(year_path):
                # Percorso della directory di destinazione per l'anno
                dest_year_path = os.path.join(dest_dir, year)
                os.makedirs(dest_year_path, exist_ok=True)

                # Itera attraverso le sottocartelle dei video
                for video_folder in os.listdir(year_path):
                    video_folder_path = os.path.join(year_path, video_folder)
                    if os.path.isdir(video_folder_path):
                        # Percorso della directory di destinazione per il video
                        dest_video_folder_path = os.path.join(dest_year_path, video_folder)
                        os.makedirs(dest_video_folder_path, exist_ok=True)

                        # Itera attraverso i file nella cartella del video
                        for file in os.listdir(video_folder_path):
                            if "_0_" in file and file.endswith(".jpg"):  # Controlla il pattern "_0_"
                                src_file_path = os.path.join(video_folder_path, file)
                                dest_file_path = os.path.join(dest_video_folder_path, file)
                                
                                # Copia il file
                                shutil.copy2(src_file_path, dest_file_path)

        print(f"Selezione completata. I frame equatoriali sono stati copiati in {dest_dir}")
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")

if __name__ == "__main__":
    # Percorsi sorgente e destinazione
    src_dir = "/home/phd2/Scrivania/CorsoData/ScopeData_extracted"
    dest_dir = "/home/phd2/Scrivania/CorsoData/ScopeData_equator"

    # Copia dei frame equatoriali
    copy_equatorial_frames(src_dir, dest_dir)
