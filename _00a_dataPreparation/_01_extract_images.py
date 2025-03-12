import sqlite3
import os
import shutil

def write_to_file(data, filename):
    """Scrive i dati binari su disco."""
    with open(filename, 'wb') as file:
        file.write(data)

def extract_frames(input_dir, output_dir, log_file, first_year:int=1900, last_year:int=3000):
    """Estrae i frame dai file .pdb mantenendo la struttura iniziale."""
    metrics = {}  # Dizionario per tracciare i video estratti e gli errori per anno
    sep = "_"
    
    try:
        for year in os.listdir(input_dir):
            year_path = os.path.join(input_dir, year)
            print(year)
            if (first_year<=int(year)<=last_year and os.path.isdir(year_path)):
                print(f"========== Estraendo anno {year} ==========")
                metrics[year] = {"videos_extracted": 0, "errors": 0}  # Inizializza le metriche per l'anno
                output_year_dir = os.path.join(output_dir, year)
                os.makedirs(output_year_dir, exist_ok=True)

                for subfolder in os.listdir(year_path):
                    subfolder_path = os.path.join(year_path, subfolder)
                    
                    if os.path.isdir(subfolder_path):
                        for file in os.listdir(subfolder_path):
                            print(file)
                            if file.endswith('.pdb'):
                                pdb_file = os.path.join(subfolder_path, file)
                                pdb_name = os.path.splitext(file)[0]

                                try:
                                    # Calculate total space needed for this PDB
                                    with sqlite3.connect(pdb_file) as con:
                                        cur = con.cursor()
                                        res = cur.execute("SELECT SUM(LENGTH(image_data)) FROM IMAGES")
                                        total_size = res.fetchone()[0] or 0
                                    
                                    # Check disk space
                                    disk_stat = shutil.disk_usage(output_dir)
                                    if disk_stat.free < total_size:
                                        print(f"Space insufficient for {pdb_file}. Skipping.")
                                        metrics[year]["errors"] += 1
                                        break
                                    
                                    # If there is enough space, continue extracting
                                    con = sqlite3.connect(pdb_file)
                                    cur = con.cursor()
                                    res = cur.execute("SELECT * FROM IMAGES")
                                    images = res.fetchall()

                                    wells = {row[0] for row in images}

                                    for well in wells:
                                        video_dir = os.path.join(output_year_dir, f"{pdb_name}{sep}{well}")
                                        os.makedirs(video_dir, exist_ok=True)
                                        metrics[year]["videos_extracted"] += 1  # Incrementa il conteggio video

                                    for row in images:
                                        well_id = row[0]
                                        timestamp = f"{row[1]}_{row[2]}_{row[3]}"
                                        image_data = row[4]

                                        image_filename = f"{pdb_name}{sep}{well_id}{sep}{timestamp}.jpg"
                                        image_path = os.path.join(output_year_dir, f"{pdb_name}{sep}{well_id}", image_filename)

                                        write_to_file(image_data, image_path)

                                    cur.close()
                                except Exception as e:
                                    print(f"========== Errore con il file {pdb_file}: {e} ==========")
                                    metrics[year]["errors"] += 1  # Incrementa il conteggio errori
            else:
                print(f"{year_path} non è una directory.")
    except Exception as e:
        print(f"Errore generale: {e}")

    # Salva i risultati in un file di log
    try:
        with open(log_file, "w") as log:
            for year, data in metrics.items():
                log.write(f"Anno: {year}\n")
                log.write(f"  Video estratti: {data['videos_extracted']}\n")
                log.write(f"  Errori: {data['errors']}\n\n")
        print(f"Risultati salvati in {log_file}")
    except Exception as e:
        print(f"Errore durante il salvataggio del log: {e}")

if __name__ == '__main__':
    # Important paths
    input_dir = "/home/phd2/Scrivania/CorsoData/ScopeData"
    output_dir = "/home/phd2/Scrivania/CorsoData/ScopeData_extracted"
    log_file = "/home/phd2/Scrivania/CorsoData/estrazione_log.txt"

    extract_frames(input_dir, output_dir, log_file)