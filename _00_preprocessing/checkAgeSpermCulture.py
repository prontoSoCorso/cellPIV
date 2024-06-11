import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, mannwhitneyu

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

def load_data(file_path):
    df = pd.read_csv(file_path, sep=',')
    df['tSB'] = df['tSB'].replace('-', np.nan).astype(float)
    return df

def plot_age_plots(df, columns, target):
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))

    for i, column in enumerate(columns):
        if column == 'maternal age':
            # Box plot per maternal age
            sns.boxplot(x=target, y=column, data=df, ax=axs[0])
            axs[0].set_title(f'Boxplot of {column} by {target}')
            
            # Aggiungi statistiche descrittive al box plot
            medians = df.groupby([target])[column].median()
            for xtick in axs[0].get_xticks():
                axs[0].text(xtick, medians[xtick] + 0.5, f'Median: {medians[xtick]:.2f}', 
                            horizontalalignment='center', size='small', color='black', weight='semibold')

            # Histogram per maternal age
            hist_plot = sns.histplot(data=df, x=column, hue=target, multiple="stack", bins=30, ax=axs[1])
            axs[1].set_title(f'Histogram of {column} by {target}')
            
            # Aggiungi i conteggi sulle barre dell'istogramma
            for p in hist_plot.patches:
                height = p.get_height()
                if height > 0:
                    axs[1].annotate(f'{height:.0f}', (p.get_x() + p.get_width() / 2., height),
                                    ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                                    textcoords='offset points')
            
            # Aggiungi mediana per ogni gruppo nell'istogramma
            for value in df[target].unique():
                median_val = df[df[target] == value][column].median()
                axs[1].axvline(median_val, color='k', linestyle='--')
                axs[1].text(median_val, max([p.get_height() for p in hist_plot.patches]),
                            f'Median: {median_val:.2f}', horizontalalignment='center', size='small',
                            color='black', weight='semibold')

    plt.tight_layout()
    plt.show()

def plot_categorical_plots(df, columns, target):
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))

    for i, column in enumerate(columns):
        if column == 'sperm quality':
            df[column] = df[column].astype('category')

            # Bar plot per sperm quality
            bar_plot = sns.countplot(x=column, hue=target, data=df, ax=axs[0])
            axs[0].set_title(f'Bar Plot of {column} by {target}')
            
            # Aggiungi i conteggi e percentuali sulle barre del barplot
            total = len(df)
            for p in bar_plot.patches:
                height = p.get_height()
                percentage = f'{100 * height / total:.1f}%'
                if height > 0:
                    axs[0].annotate(f'{height:.0f}\n({percentage})', (p.get_x() + p.get_width() / 2., height),
                                    ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                                    textcoords='offset points')

        elif column == 'mezzo di coltura':
            df[column] = df[column].astype('category')

            # Bar plot per mezzo di coltura
            bar_plot = sns.countplot(x=column, hue=target, data=df, ax=axs[1])
            axs[1].set_title(f'Bar Plot of {column} by {target}')
            
            # Aggiungi i conteggi e percentuali sulle barre del barplot
            for p in bar_plot.patches:
                height = p.get_height()
                percentage = f'{100 * height / total:.1f}%'
                if height > 0:
                    axs[1].annotate(f'{height:.0f}\n({percentage})', (p.get_x() + p.get_width() / 2., height),
                                    ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                                    textcoords='offset points')

    plt.tight_layout()
    plt.show()

def chi_square_test(df, column, target):
    contingency_table = pd.crosstab(df[column], df[target])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p

def mann_whitney_test(df, column, target):
    group1 = df[df[target] == 0][column].dropna()
    group2 = df[df[target] == 1][column].dropna()
    stat, p = mannwhitneyu(group1, group2)
    return stat, p

if __name__ == '__main__':
    df = load_data(input_csv_path)

    # Variabili da analizzare
    age_columns = ['maternal age']
    categorical_columns = ['sperm quality', 'mezzo di coltura']
    target_column = 'BLASTO NY'

    plot_age_plots(df, age_columns, target_column)
    plot_categorical_plots(df, categorical_columns, target_column)

    # Test statistici
    for column in age_columns + categorical_columns:
        if df[column].dtype.name == 'category':
            chi2, p = chi_square_test(df, column, target_column)
            print(f'Chi-square test for {column}: chi2 = {chi2}, p = {p}')
        else:
            stat, p = mann_whitney_test(df, column, target_column)
            print(f'Mann-Whitney U test for {column}: stat = {stat}, p = {p}')


'''
Mann-Whitney U test for maternal age: stat = 3085033.0, p = 4.770731369086512e-13
Chi-square test for sperm quality: chi2 = 14.045403938998533, p = 0.23049126745433737
Chi-square test for mezzo di coltura: chi2 = 53.11968823052976, p = 2.9188129237579764e-12

Nel caso dell'eta della donna la differenza è molto significativa e potrebbe essere una variabile di confondimento
Nel caso della qualità dello sperma la differenza non è statisticamente significativa.
Nel caso del mezzo di coltura la differenza è molto significativa e potrebbe essere una variabile di confondimento
'''
