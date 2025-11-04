import csv

# Definisci i percorsi dei file e le etichette corrispondenti
file_paths = {
    'train': '/home/phd2/Scrivania/CorsoRepo/cellPIV/datasets/Farneback/subsets/Normalized_sum_mean_mag_1Days_train.csv',
    'test': '/home/phd2/Scrivania/CorsoRepo/cellPIV/datasets/Farneback/subsets/Normalized_sum_mean_mag_1Days_test.csv',
    'val': '/home/phd2/Scrivania/CorsoRepo/cellPIV/datasets/Farneback/subsets/Normalized_sum_mean_mag_1Days_val.csv'
}

output_file = '/home/phd2/Scrivania/CorsoRepo/cellPIV/datasets/splits.txt'

with open(output_file, 'w') as outfile:
    for split_type, file_path in file_paths.items():
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dish_well = row['dish_well']
                outfile.write(f"{dish_well},{split_type}\n")

print(f"File creato con successo: {output_file}")