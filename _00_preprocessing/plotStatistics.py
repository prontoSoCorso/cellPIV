import pandas as pd
import matplotlib.pyplot as plt
import os

# Dati dell'output
samples_per_year = {'2011': 1, '2013': 321, '2014': 711, '2015': 635, '2016': 607, '2017': 620, '2018': 692, '2019': 660, '2020': 559}
blastocysts_per_year = {'2011': 1, '2013': 160, '2014': 424, '2015': 364, '2016': 361, '2017': 348, '2018': 392, '2019': 415, '2020': 467}
non_blastocysts_per_year = {'2013': 161, '2014': 287, '2015': 271, '2016': 246, '2017': 272, '2018': 300, '2019': 245, '2020': 92}

# Convertiamo i dati in dataframe per facilitare la creazione del grafico
df_samples_per_year = pd.DataFrame(list(samples_per_year.items()), columns=['Year', 'Samples'])
df_blastocysts_per_year = pd.DataFrame(list(blastocysts_per_year.items()), columns=['Year', 'Blastocysts'])
df_non_blastocysts_per_year = pd.DataFrame(list(non_blastocysts_per_year.items()), columns=['Year', 'Non-Blastocysts'])

# Calcolare le percentuali
df_samples_per_year['Percentage'] = df_samples_per_year['Samples'] / df_samples_per_year['Samples'].sum() * 100
df_blastocysts_per_year['Percentage'] = df_blastocysts_per_year['Blastocysts'] / df_blastocysts_per_year['Blastocysts'].sum() * 100
df_non_blastocysts_per_year['Percentage'] = df_non_blastocysts_per_year['Non-Blastocysts'] / df_non_blastocysts_per_year['Non-Blastocysts'].sum() * 100

# Prima figura: Samples e Blastocysts
fig1, axs1 = plt.subplots(2, 1, figsize=(10, 10))
fig1.suptitle('Summary of Samples and Blastocysts Data', fontsize=16)

# Impostare lo stesso limite y per i bar plot
max_y1 = max(df_samples_per_year['Samples'].max(), df_blastocysts_per_year['Blastocysts'].max())

# Numero di campioni per anno
bars1 = axs1[1].bar(df_samples_per_year['Year'], df_samples_per_year['Samples'], color='blue', alpha=0.7)
axs1[1].set_title('Number of Samples per Year')
axs1[1].set_xlabel('Year')
axs1[1].set_ylabel('Number of Samples')
axs1[1].set_ylim(0, max_y1 * 1.1)
for bar, percentage in zip(bars1, df_samples_per_year['Percentage']):
    axs1[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{percentage:.2f}%', ha='center', va='bottom')

# Numero di blastocisti per anno
bars1 = axs1[0].bar(df_blastocysts_per_year['Year'], df_blastocysts_per_year['Blastocysts'], color='green', alpha=0.7)
axs1[0].set_title('Number of Blastocysts per Year')
axs1[0].set_xlabel('Year')
axs1[0].set_ylabel('Number of Blastocysts')
axs1[0].set_ylim(0, max_y1 * 1.1)
for bar, percentage in zip(bars1, df_blastocysts_per_year['Percentage']):
    axs1[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{percentage:.2f}%', ha='center', va='bottom')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Seconda figura: Non-Blastocysts e Testi
fig2, axs2 = plt.subplots(2, 1, figsize=(10, 10))
fig2.suptitle('Summary of Samples and Blastocysts Data', fontsize=16)

# Numero di non-blastocisti per anno
bars2 = axs2[0].bar(df_non_blastocysts_per_year['Year'], df_non_blastocysts_per_year['Non-Blastocysts'], color='red', alpha=0.7)
axs2[0].set_title('Number of Non-Blastocysts per Year')
axs2[0].set_xlabel('Year')
axs2[0].set_ylabel('Number of Non-Blastocysts')
axs2[0].set_ylim(0, max_y1 * 1.1)
for bar, percentage in zip(bars2, df_non_blastocysts_per_year['Percentage']):
    axs2[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{percentage:.2f}%', ha='center', va='bottom')

# Dati dell'output
num_samples = 4806
num_blastocysts = 2932
num_non_blastocysts = 1874
samples_2013_2020 = 4805
blastocysts_2013_2020 = 2931
non_blastocysts_2013_2020 = 1874

# Creazione del grafico con i testi
axs2[1].axis('off')
textstr = '\n'.join((
    f'Number of samples: {num_samples}',
    f'Number of blastocysts: {num_blastocysts}',
    f'Number of non-blastocysts: {num_non_blastocysts}',
    f'Number of samples from 2013 to 2020: {samples_2013_2020}',
    f'Number of blastocysts from 2013 to 2020: {blastocysts_2013_2020}',
    f'Number of non-blastocysts from 2013 to 2020: {non_blastocysts_2013_2020}'
))

# Aggiungi il testo al grafico
axs2[1].text(0.5, 0.5, textstr, transform=axs2[1].transAxes, fontsize=14,
             verticalalignment='center', horizontalalignment='center', weight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
