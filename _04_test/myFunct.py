import matplotlib.pyplot as plt
import seaborn as sns

# Funzione per salvare la matrice di contingenza con il risultato di McNemar
def save_contingency_matrix_with_mcnemar(matrix, filename, model_1_name, model_2_name, p_value):
    plt.figure(figsize=(6, 6))  # Riduci la dimensione della matrice
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["Model 2 Wrong", "Model 2 Correct"],
                yticklabels=["Model 1 Wrong", "Model 1 Correct"],
                annot_kws={"fontsize": 12})  # Aumenta dimensione dei numeri annotati

    # Aumenta la dimensione dei font per gli assi
    plt.xlabel(f"{model_1_name} Predictions", fontsize=15, labelpad=20)
    plt.ylabel(f"{model_2_name} Predictions", fontsize=15, labelpad=20)

    # Aumenta dimensione xticks e yticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.title("\nContingency Matrix with McNemar Test\n", fontsize=16)  # Aumenta titolo

    # Mostra il p-value con un carattere più grande
    plt.figtext(0.5, -0.1, f"McNemar Test p-value: {p_value:.2e}",
                ha="center", fontsize=14, wrap=True, bbox={"facecolor": "lightgrey", "alpha": 0.5, "pad": 5})
    
    plt.savefig(filename, bbox_inches="tight")
    plt.close()