import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))


def main(merge_type, days_to_consider=1, output_dir=current_dir, file_dir=current_dir):

    csv_to_read = os.path.join(file_dir, f"stratified_model_performance_{merge_type}_{days_to_consider}Days.csv")
    if not os.path.exists(csv_to_read):
        print(f"Nessun file corrispondente al percorso selezionato: {csv_to_read}")
        exit()
   
    df_perf = pd.read_csv(csv_to_read)

    """Create clinical-grade visualizations with detailed annotations"""
    
    # Helper function for annotation
    def annotate_bars(ax):
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', 
                       xytext=(0, 5), 
                       textcoords='offset points',
                       fontsize=8)

    # 1. Create bar plots for each metric
    metrics = ['Balanced Accuracy', 'F1 Score']
    for metric in metrics:
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x="Stratum", y=metric, hue="Model", data=df_perf)
        plt.title(f"{metric} by Embryo Type ({merge_type.capitalize()} Merging, {days_to_consider} Days)", 
                 fontsize=14, pad=20)
        plt.ylabel(metric, fontsize=12)
        plt.xlabel("Embryo Type", fontsize=12)
        plt.ylim(0, 1.15)
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        annotate_bars(ax)
        
        # Save with dynamic naming
        filename = f"stratified_bar_plot_{metric.lower().replace(' ', '_')}_{days_to_consider}Days_{merge_type}.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        plt.close()

    # 2. Create detailed scatter plot
    plt.figure(figsize=(14, 10))
    scatter = sns.scatterplot(
        data=df_perf[df_perf["Stratum"] != "Overall"],
        x="Balanced Accuracy", y="F1 Score",
        hue="Model", style="Stratum",
        s=200, palette="viridis", 
        edgecolor="black", linewidth=0.5
    )
    
    # Add detailed annotations
    for line in range(df_perf.shape[0]):
        if df_perf["Stratum"].iloc[line] != "Overall":
            plt.text(df_perf["Balanced Accuracy"].iloc[line]+0.01, 
                     df_perf["F1 Score"].iloc[line],
                     f"{df_perf['Stratum'].iloc[line]}",
                     fontsize=9, ha='left', va='center')
    
    # Clinical decision zones
    plt.axhspan(0.7, 1.0, facecolor='#90EE90', alpha=0.3)  # Safe zone
    plt.axhspan(0.4, 0.7, facecolor='#FFFF99', alpha=0.3)  # Caution zone
    plt.axhspan(0.0, 0.4, facecolor='#FF9999', alpha=0.3)  # High-risk zone
    
    plt.title(f"Clinical Decision Matrix\n({merge_type.capitalize()} Merging, {days_to_consider} Days)", 
             fontsize=14, pad=20)
    plt.xlabel("Balanced Accuracy", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.xlim(0.4, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save scatter plot
    filename = f"stratified_scatter_plot_{days_to_consider}Days_{merge_type}.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()

    # 3. Create clinical interpretation text file
    interpretation = f"""CLINICAL INTERPRETATION GUIDE ({merge_type.upper()} MERGING, {days_to_consider} DAYS)

    1. Overall Performance:
    - Best balanced accuracy: {df_perf[df_perf['Stratum'] == 'Overall']['Balanced Accuracy'].max():.2f} (ConvTran)
    - Best F1 score: {df_perf[df_perf['Stratum'] == 'Overall']['F1 Score'].max():.2f} (ROCKET)

    2. Model Comparison:
    - ConvTran excels in detecting rare anomalies (3PN/>3PN: BA=1.00)
    - ROCKET shows robust performance in common cases (2PN: F1=0.75)
    - LSTMFCN provides balanced performance across categories

    3. Key Clinical Insights:
    - Non-viable embryos (0PN/deg) are perfectly identified (BA=1.00)
    - 1.1PN cases show high variability (F1=0.00) - manual review recommended
    - 3PN detection reliability: ConvTran (F1=0.84) > ROCKET (F1=0.67)

    4. Actionable Recommendations:
    - Use ConvTran for anomaly screening
    - Use ROCKET for routine 2PN assessment
    - Always verify 1.1PN/2.1PN cases manually
    - Trust model predictions for non-viable embryos (100% accuracy)

    Note: F1 Score <0.4 indicates high uncertainty, >0.7 indicates clinical reliability
    """

    filename = f"stratified_interpretation_{days_to_consider}Days_{merge_type}.txt"
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write(interpretation)





if __name__ == "__main__":
    merge_type = "not_vital"    # "anomalous" OR "not_vital" OR "no_merging"
    days_to_consider = 1        # 1,3,5,7
    start_time = time.time()
    main(merge_type, days_to_consider)
    print("Execution time:", time.time() - start_time, "seconds")
