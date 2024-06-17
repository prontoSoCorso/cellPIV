import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, kstest, shapiro, mannwhitneyu, normaltest, probplot, ttest_ind, pointbiserialr

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
    #plt.show()

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
    #plt.show()

def normality_tests(df, column, target):
    group1 = df[df[target] == 0][column].dropna()
    group2 = df[df[target] == 1][column].dropna()

    # Kolmogorov-Smirnov test per ogni gruppo
    ks_stat1, ks_p1 = kstest((group1-group1.mean())/group1.std(), 'norm')
    ks_stat2, ks_p2 = kstest((group2-group2.mean())/group2.std(), 'norm')
    
    print(f"Kolmogorov-Smirnov test for group1: statistic={ks_stat1:.4f}, p-value={ks_p1:.4f}")
    print(f"Kolmogorov-Smirnov test for group2: statistic={ks_stat2:.4f}, p-value={ks_p2:.4f}")

    # Shapiro-Wilk test per ogni gruppo
    sw_stat1, sw_p1 = shapiro(group1)
    sw_stat2, sw_p2 = shapiro(group2)
    
    print(f"Shapiro-Wilk test for group1: statistic={sw_stat1:.4f}, p-value={sw_p1:.4f}")
    print(f"Shapiro-Wilk test for group2: statistic={sw_stat2:.4f}, p-value={sw_p2:.4f}")

    # D'Agostino and Pearson's test per ogni gruppo
    dp_stat1, dp_p1 = normaltest(group1)
    dp_stat2, dp_p2 = normaltest(group2)

    print(f"D'Agostino and Pearson's test for group1: statistic={dp_stat1:.4f}, p-value={dp_p1:.4f}")
    print(f"D'Agostino and Pearson's test for group2: statistic={dp_stat2:.4f}, p-value={dp_p2:.4f}")

    # QQ plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    probplot(group1, dist="norm", plot=axs[0])
    axs[0].set_title(f'QQ Plot for {column} of group1')
    
    probplot(group2, dist="norm", plot=axs[1])
    axs[1].set_title(f'QQ Plot for {column} of group2')
    
    #plt.show()


def mann_whitney_u_test_with_effect_size(df, column, target):
    group1 = df[df[target] == 0][column].dropna()
    group2 = df[df[target] == 1][column].dropna()
    u_stat, u_p = mannwhitneyu(group1, group2)

    # Calcolo della statistica Z
    n1 = len(group1)
    n2 = len(group2)
    mean_rank = (n1 + n2 + 1) / 2.0
    std_rank = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12.0)
    z_stat = (u_stat - (n1 * n2 / 2.0)) / std_rank
    
    # Calcolo dell'effect size r
    effect_size_r = z_stat / np.sqrt(n1 + n2)
    
    return u_stat, u_p, effect_size_r


def student_t_test_with_effect_size(df, column, target):
    group1 = df[df[target] == 0][column].dropna()
    group2 = df[df[target] == 1][column].dropna()
    t_stat, p_value = ttest_ind(group1, group2)

    # Calcolo del Cohen's d effect size
    mean1 = group1.mean()
    mean2 = group2.mean()
    std_pooled = np.sqrt(((group1.var() + group2.var()) / 2))
    cohens_d = (mean1 - mean2) / std_pooled

    return t_stat, p_value, cohens_d


def chi_square_test_with_effect_size(df, column, target):
    contingency_table = pd.crosstab(df[column], df[target])
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Calcolo di Cramér's V
    n = contingency_table.sum().sum()
    k = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k))

    return chi2, p, cramers_v

def point_biserial_correlation(df, column, target):
    correlation, p_value = pointbiserialr(df[target], df[column].dropna())
    return correlation, p_value

if __name__ == '__main__':
    df = load_data(input_csv_path)

    # Variabili da analizzare
    continuous_columns = ['maternal age']
    categorical_columns = ['sperm quality', 'mezzo di coltura']
    target_column = 'BLASTO NY'

    plot_age_plots(df, continuous_columns, target_column)
    plot_categorical_plots(df, categorical_columns, target_column)

    # Test di normalità per "maternal age"
    for column in continuous_columns:
        print(f"\nNormality tests for {column}:")
        normality_tests(df, column, target_column)

    # Test statistici
    for column in continuous_columns:
        # Usare Mann-Whitney U Test invece del t-test con effect size
        u_stat, u_p, effect_size_r = mann_whitney_u_test_with_effect_size(df, column, target_column)
        print(f'\nMann-Whitney U Test for {column}: U-statistic = {u_stat}, p-value = {u_p}, effect size r = {effect_size_r}')
        
        '''
        # Usare t-Test con effect size, NO PERCHE' DISTRIBUZIONE NON GAUSSIANA
        t_stat, t_p, cohens_d = student_t_test_with_effect_size(df, column, target_column)
        print(f't-Test for {column}: t-statistic = {t_stat}, p-value = {t_p}, effect size d = {cohens_d}')
        '''

        # Point-Biserial Correlation Coefficient
        correlation, p_value = point_biserial_correlation(df, column, target_column)
        print(f'Point-Biserial Correlation for {column}: p-value = {p_value}, correlation = {correlation}')

    for column in categorical_columns:
        chi2, p, cramers_v = chi_square_test_with_effect_size(df, column, target_column)
        print(f'Chi-square test for {column}: chi2 = {chi2}, p = {p}, Cramér\'s V = {cramers_v}')





'''
Nel caso dell'eta della donna la differenza è molto significativa e potrebbe essere una variabile di confondimento
Nel caso della qualità dello sperma la differenza non è statisticamente significativa.
Nel caso del mezzo di coltura la differenza è molto significativa e potrebbe essere una variabile di confondimento

Possibile spiegazione risultati:
- Il t-test confronta le medie di due gruppi e, con campioni grandi, anche piccole differenze tra i gruppi possono 
risultare statisticamente significative se la variabilità all'interno dei gruppi è bassa.
Nel caso di questi dati, non essendo distribuiti normalmente (o almeno, questo dicono i test statistici), 
non uso il t-test ma il mann-whitney U, ma le idee di base sono le stesse (troppi campioni forse)

- Con campioni molto grandi, i test di normalità hanno una potenza elevata e possono rilevare deviazioni minime 
dalla normalità che non sono visibili nei QQ plot.


Per questi motivi, oltre a considerare il p-value, considero anche altri parametri, come l'effect-size.
Per il Mann-Whitney U test, una misura comune dell'effect size è Cohen's d o r. L'effect size rr per il 
Mann-Whitney U test può essere calcolato come segue: r = Z/sqrt(N)
Z è la statistica Z trasformata dalla statistica U. N è il numero totale di osservazioni.

Considero in generale la divisione:
0.1: Small effect
0.3: Medium effect
0.5: Large effect
Mentre il p-value ti dice se c'è una differenza statisticamente significativa tra i gruppi, l'effect size ti dice 
quanto è grande questa differenza, che è particolarmente utile per interpretare i risultati in studi con grandi 
campioni.

Infine, considero la point-biserial correlation per determinare se la correlazione tra maternal age ed output è forte.
La correlazione point-biseriale è una misura di correlazione che si applica quando si ha una variabile dicotomica (binaria) 
e una variabile continua. È una versione speciale del coefficiente di correlazione di Pearson.

Il risultato è che l'età materna ha un effetto minimo sull'output "BLASTO NY" e non dovrebbe essere considerata un 
fattore determinante principale per l'output. 

NB: Entrambe le misure forniscono informazioni utili ma complementari: la correlazione dice quanto sono linearmente associate
    le variabili, mentre l'effect size dice quanto grande è la differenza osservata o l'effetto nel contesto dello studio.



Per le variabili categoriche, una misura comune dell'effect size è Cramér's V. Questo valore è derivato dalla 
statistica Chi-quadro e fornisce una misura della forza dell'associazione tra le variabili categoriche

V=sqrt( (chi^2)/ N⋅(k-1) )

Posso fare un ragionamento simile anche per il kstest, anche se non fornisce un effetto delle dimensioni 
diretto come Cramér's V. Tuttavia possiamo considerare il valore D del KS test rappresenta la distanza massima 
tra la funzione di distribuzione empirica (ECDF) e la funzione di distribuzione cumulativa (CDF) teorica. 
Anche se non è un vero e proprio effetto delle dimensioni, il valore D può essere utilizzato per avere un'idea 
dell'effetto.

Linee guida:
    0 ≤ D < 0.1: Piccola differenza (molto vicina alla distribuzione teorica).
    0.1 ≤ D < 0.2: Differenza moderata.
    D ≥ 0.2: Grande differenza (significativa deviazione dalla distribuzione teorica).

    



OUTPUTS:
Normality tests for maternal age:
Kolmogorov-Smirnov test for group1: statistic=0.1064, p-value=0.0000
Kolmogorov-Smirnov test for group2: statistic=0.0781, p-value=0.0000
Shapiro-Wilk test for group1: statistic=0.9674, p-value=0.0000
Shapiro-Wilk test for group2: statistic=0.9735, p-value=0.0000
D'Agostino and Pearson's test for group1: statistic=105.5416, p-value=0.0000
D'Agostino and Pearson's test for group2: statistic=157.7006, p-value=0.0000

Mann-Whitney U Test for maternal age: U-statistic = 3085033.0, p-value = 4.770731369086512e-13, effect size r = 0.10384580907290734     
Point-Biserial Correlation for maternal age: p-value = 1.3478907145812641e-11, correlation = -0.09735384901324291
Chi-square test for sperm quality: chi2 = 14.045403938998533, p = 0.23049126745433737, Cramér's V = 0.05405989946080693
Chi-square test for mezzo di coltura: chi2 = 53.11968823052976, p = 2.9188129237579764e-12, Cramér's V = 0.10513222975152786

'''