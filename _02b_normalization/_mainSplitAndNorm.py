import os
import sys
import time

# Rileva il percorso della cartella genitore, che sarà la stessa in cui ho il file da convertire
current_dir = os.path.dirname(os.path.abspath(__file__))

# Individua la cartella 'cellPIV' come riferimento
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while os.path.basename(parent_dir) != "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_02b_normalization as conf
from config import utils
from _02b_normalization._01_split_normalization import load_data, normalize_data, save_data, stratified_split
from _02b_normalization._02_visualization import visualize_normalized_data, create_and_save_plots
from _utils_.dimReduction import compute_UMAP, compute_tSNE



def main(days_to_consider=conf.days_to_consider, train_size=conf.train_size, seed=conf.seed, embedding_type=conf.embedding_type, save_normalization_example=True):
    # Carica i dati
    max_frames = utils.num_frames_by_days(days_to_consider)
    csv_file_path = conf.csv_file_path
    data = load_data(csv_file_path=csv_file_path, max_frames=max_frames)

    # Split stratificato
    train_data, val_data, test_data = stratified_split(data, train_size=train_size, seed=seed)

    # Normalizza i dati
    train_data_norm, val_data_norm, test_data_norm = normalize_data(train_data, val_data, test_data)

    # Salva i dataset normalizzati
    base_path = conf.get_normalized_base_path(days_to_consider)
    save_data(train_data=train_data_norm, val_data=val_data_norm, test_data=test_data_norm, output_base_path=base_path, days_to_consider=days_to_consider)

    # Visualizzo embedding di un subset a scelta --> ottengo il percorso, ad esempio, del train (numero maggiore di dati)
    train_path, val_path, test_path = conf.get_paths(days_to_consider)
    output_path_base = os.path.join(current_dir, "dim_reduction_files")
    os.makedirs(output_path_base, exist_ok=True)
    max_frames = utils.num_frames_by_days(days_to_consider)

    if embedding_type:
        if embedding_type.lower()=="umap":
            print("Computing UMAP...")
            compute_UMAP(csv_path=train_path, days_to_consider=days_to_consider, max_frames=max_frames, output_path_base=output_path_base)
        elif embedding_type.lower()=="tsne":
            print("Computing tSNE...")
            compute_tSNE(csv_path=train_path, days_to_consider=days_to_consider, max_frames=max_frames, output_path_base=output_path_base)
        else:
            print("Please select a valid dimensionality reduction method. You can choose from: umap, tsne")

    if save_normalization_example:
        visualize_normalized_data(
            original_data=train_data,
            normalized_data=train_data_norm,
            output_base=current_dir
        )

        temporalDataType = conf.temporalDataType
        create_and_save_plots(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        output_base=current_dir,
        seed=seed,
        temporal_data_type=temporalDataType,
        days_to_consider=days_to_consider
        )


if __name__ == "__main__":
    start_time = time.time()
    main(days_to_consider=3, train_size=0.7, seed=42, embedding_type="", save_normalization_example=True)
    print("Execution time: ", str(time.time()-start_time), "seconds")
