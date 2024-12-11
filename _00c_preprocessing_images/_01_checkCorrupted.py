import os
from PIL import Image

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

if __name__ == "__main__":
    # Percorso principale delle cartelle
    root_dir = "/home/phd2/Scrivania/CorsoData/blastocisti"

    # Percorso per il file di log
    log_file_path = os.path.join(root_dir, "corrupted_images_log.txt")

    # Controlla le immagini
    check_for_corrupted_images(root_dir, log_file_path)




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
