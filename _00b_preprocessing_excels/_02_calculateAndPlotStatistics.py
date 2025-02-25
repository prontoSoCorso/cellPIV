import pandas as pd
import numpy as np
import os
from tabulate import tabulate
import matplotlib.pyplot as plt

# Funzioni di utilità
def load_data(file_path):
    df = pd.read_csv(file_path, sep=',')
    df['tSB'] = df['tSB'].replace('-', np.nan).astype(float)
    return df

def calculate_n_images(df):
    df['n_images'] = df.apply(lambda row: int(row['tSB'] * 4) if row['BLASTO NY'] == 1 and not pd.isnull(row['tSB']) else 0, axis=1)
    return df

def extract_year_from_dish(df):
    df['year'] = df['dish'].str.split('.').str[0].str.split('D').str[-1]
    return df

def num_samples(df):
    return df.shape[0]

def num_blastocysts(df):
    return df[df['BLASTO NY'] == 1].shape[0]

def num_non_blastocysts(df):
    return df[df['BLASTO NY'] == 0].shape[0]

def num_samples_per_year(df):
    df = extract_year_from_dish(df)
    return df.groupby('year').size().to_dict()

def num_blastocysts_per_year(df):
    df = extract_year_from_dish(df)
    return df[df['BLASTO NY'] == 1].groupby('year').size().to_dict()

def num_non_blastocysts_per_year(df):
    df = extract_year_from_dish(df)
    return df[df['BLASTO NY'] == 0].groupby('year').size().to_dict()

def save_plot_images(samples_per_year, blastocysts_per_year, non_blastocysts_per_year, summary_text, output_dir):
    df_samples = pd.DataFrame(list(samples_per_year.items()), columns=['Year', 'Samples'])
    df_blastocysts = pd.DataFrame(list(blastocysts_per_year.items()), columns=['Year', 'Blastocysts'])
    df_non_blastocysts = pd.DataFrame(list(non_blastocysts_per_year.items()), columns=['Year', 'Non-Blastocysts'])

    df_samples['Percentage'] = df_samples['Samples'] / df_samples['Samples'].sum() * 100
    df_blastocysts['Percentage'] = df_blastocysts['Blastocysts'] / df_blastocysts['Blastocysts'].sum() * 100
    df_non_blastocysts['Percentage'] = df_non_blastocysts['Non-Blastocysts'] / df_non_blastocysts['Non-Blastocysts'].sum() * 100

    fig1, axs1 = plt.subplots(2, 1, figsize=(10, 10))
    fig1.suptitle('Summary of Samples and Blastocysts Data', fontsize=16)

    max_y = max(df_samples['Samples'].max(), df_blastocysts['Blastocysts'].max())

    bars1 = axs1[1].bar(df_samples['Year'], df_samples['Samples'], color='blue', alpha=0.7)
    axs1[1].set_title('Number of Samples per Year')
    axs1[1].set_xlabel('Year')
    axs1[1].set_ylabel('Number of Samples')
    axs1[1].set_ylim(0, max_y * 1.1)
    for bar, percentage in zip(bars1, df_samples['Percentage']):
        axs1[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{percentage:.2f}%', ha='center', va='bottom')

    bars1 = axs1[0].bar(df_blastocysts['Year'], df_blastocysts['Blastocysts'], color='green', alpha=0.7)
    axs1[0].set_title('Number of Blastocysts per Year')
    axs1[0].set_xlabel('Year')
    axs1[0].set_ylabel('Number of Blastocysts')
    axs1[0].set_ylim(0, max_y * 1.1)
    for bar, percentage in zip(bars1, df_blastocysts['Percentage']):
        axs1[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{percentage:.2f}%', ha='center', va='bottom')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "plots_statistics1.png"), dpi=600)

    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 10))
    fig2.suptitle('Summary of Non-Blastocysts and Text Data', fontsize=16)

    bars2 = axs2[0].bar(df_non_blastocysts['Year'], df_non_blastocysts['Non-Blastocysts'], color='red', alpha=0.7)
    axs2[0].set_title('Number of Non-Blastocysts per Year')
    axs2[0].set_xlabel('Year')
    axs2[0].set_ylabel('Number of Non-Blastocysts')
    axs2[0].set_ylim(0, max_y * 1.1)
    for bar, percentage in zip(bars2, df_non_blastocysts['Percentage']):
        axs2[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{percentage:.2f}%', ha='center', va='bottom')

    axs2[1].axis('off')
    axs2[1].text(0.5, 0.5, summary_text, transform=axs2[1].transAxes, fontsize=14, verticalalignment='center', horizontalalignment='center', weight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "plots_statistics2.png"), dpi=600)


def calculate_and_plot_statistics(input_csv_path, output_dir):
    df = load_data(input_csv_path)
    df = calculate_n_images(df)

    data_summary = [
        ['Total Samples', num_samples(df)],
        ['Blastocysts', num_blastocysts(df)],
        ['Non-Blastocysts', num_non_blastocysts(df)]
    ]

    yearly_summary = []
    yearly_data = num_samples_per_year(df)
    blastocysts_data = num_blastocysts_per_year(df)
    non_blastocysts_data = num_non_blastocysts_per_year(df)

    for year in sorted(yearly_data.keys()):
        yearly_summary.append([
            year,
            yearly_data.get(year, 0),
            blastocysts_data.get(year, 0),
            non_blastocysts_data.get(year, 0)
        ])

    print("Overall Summary:")
    print(tabulate(data_summary, headers=['Metric', 'Value'], tablefmt='grid'))

    print("\nYearly Summary:")
    print(tabulate(yearly_summary, headers=['Year', 'Samples', 'Blastocysts', 'Non-Blastocysts'], tablefmt='grid'))

    summary_text = '\n'.join([
        f"Number of samples: {num_samples(df)}",
        f"Number of blastocysts: {num_blastocysts(df)}",
        f"Number of non-blastocysts: {num_non_blastocysts(df)}"
    ])

    save_plot_images(yearly_data, blastocysts_data, non_blastocysts_data, summary_text, output_dir)
