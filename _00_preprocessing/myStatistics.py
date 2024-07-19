import pandas as pd
import numpy as np
import os
import sys

# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_00_preprocessing as conf

# Percorso del file CSV
input_csv_path = conf.path_new_excel

# Funzioni di utilità
def load_data(file_path):
    df = pd.read_csv(file_path, sep=',')
    df['tSB'] = df['tSB'].replace('-', np.nan).astype(float)
    return df

def calculate_n_images(df):
    df['n_images'] = df.apply(lambda row: int(row['tSB'] * 4) if row['BLASTO NY'] == 1 and not pd.isnull(row['tSB']) else 0, axis=1)
    return df

def extract_year_from_slide(df):
    df['year'] = df['slide'].str.split('.').str[0].str.split('D').str[-1]
    return df

def num_samples(df):
    return df.shape[0]

def num_blastocysts(df):
    return df[df['BLASTO NY'] == 1].shape[0]

def num_non_blastocysts(df):
    return df[df['BLASTO NY'] == 0].shape[0]

def num_samples_per_year(df):
    df = extract_year_from_slide(df)
    return df.groupby('year').size()

def num_blastocysts_per_year(df):
    df = extract_year_from_slide(df)
    return df[df['BLASTO NY'] == 1].groupby('year').size()

def num_non_blastocysts_per_year(df):
    df = extract_year_from_slide(df)
    return df[df['BLASTO NY'] == 0].groupby('year').size()

def num_samples_from_year_to_year(df, year1, year2):
    df = extract_year_from_slide(df)
    return df[(df['year'] >= year1) & (df['year'] <= year2)].shape[0]

def num_blastocysts_from_year_to_year(df, year1, year2):
    df = extract_year_from_slide(df)
    return df[(df['year'] >= year1) & (df['year'] <= year2) & (df['BLASTO NY'] == 1)].shape[0]

def num_non_blastocysts_from_year_to_year(df, year1, year2):
    df = extract_year_from_slide(df)
    return df[(df['year'] >= year1) & (df['year'] <= year2) & (df['BLASTO NY'] == 0)].shape[0]

if __name__ == '__main__':
    df = load_data(input_csv_path)
    df = calculate_n_images(df)

    print('Number of samples: {}'.format(num_samples(df)))
    print('Number of blastocysts: {}'.format(num_blastocysts(df)))
    print('Number of non-blastocysts: {}'.format(num_non_blastocysts(df)))
    print('Number of samples per year: \n{}'.format(num_samples_per_year(df)))
    print('Number of blastocysts per year: \n{}'.format(num_blastocysts_per_year(df)))
    print('Number of non-blastocysts per year: \n{}'.format(num_non_blastocysts_per_year(df)))
    print('Number of samples from 2013 to 2020: {}'.format(num_samples_from_year_to_year(df, '2013', '2020')))
    print('Number of blastocysts from 2013 to 2020: {}'.format(num_blastocysts_from_year_to_year(df, '2013', '2020')))
    print('Number of non-blastocysts from 2013 to 2020: {}'.format(num_non_blastocysts_from_year_to_year(df, '2013', '2020')))

    