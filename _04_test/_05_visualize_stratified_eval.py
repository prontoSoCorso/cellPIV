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

    # Use a more distinct color palette
    model_palette = {
        "ROCKET": "#1f77b4",  # Distinct blue
        "LSTMFCN": "#ff7f0e",  # Bright orange
        "ConvTran": "#2ca02c"  # Strong green
    }

    # Create scatter plot with enhanced visibility
    scatter = sns.scatterplot(
        data=df_perf[df_perf["Stratum"] != "Overall"],
        x="Balanced Accuracy", y="F1 Score",
        hue="Model", style="Stratum",
        s=200, palette=model_palette,
        edgecolor="black", linewidth=0.8,  # Thicker borders
        alpha=0.9  # Slightly more opaque
    )

    # Add detailed annotations with better contrast
    for line in range(df_perf.shape[0]):
        if df_perf["Stratum"].iloc[line] != "Overall":
            plt.text(df_perf["Balanced Accuracy"].iloc[line] + 0.015,  # Slightly more offset
                    df_perf["F1 Score"].iloc[line],
                    f"{df_perf['Stratum'].iloc[line]}",
                    fontsize=10, ha='left', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    # Enhanced clinical decision zones
    plt.axhspan(0.7, 1.0, facecolor='#90EE90', alpha=0.25)  # Safe zone
    plt.axhspan(0.4, 0.7, facecolor='#FFFF99', alpha=0.25)  # Caution zone
    plt.axhspan(0.0, 0.4, facecolor='#FF9999', alpha=0.25)  # High-risk zone

    # Add zone labels
    plt.text(0.45, 0.85, "Safe Zone", fontsize=12, color='darkgreen', alpha=0.8)
    plt.text(0.45, 0.55, "Caution Zone", fontsize=12, color='darkgoldenrod', alpha=0.8)
    plt.text(0.45, 0.25, "High-Risk Zone", fontsize=12, color='darkred', alpha=0.8)

    # Plot formatting
    plt.title(f"Clinical Decision Matrix\n({merge_type.capitalize()} Merging, {days_to_consider} Days)", 
            fontsize=16, pad=20, weight='bold')
    plt.xlabel("Balanced Accuracy", fontsize=14)
    plt.ylabel("F1 Score", fontsize=14)
    plt.xlim(0.4, 1.05)
    plt.ylim(-0.05, 1.05)

    # Enhanced legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
            frameon=True, framealpha=0.9,
            title="Model/Stratum", title_fontsize=12,
            fontsize=10, markerscale=1.5)

    # Save scatter plot
    filename = f"stratified_scatter_plot_{days_to_consider}Days_{merge_type}.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    merge_type = "not_vital"    # "anomalous" OR "not_vital" OR "no_merging"
    days_to_consider = 5        # 1,3,5,7
    start_time = time.time()
    main(merge_type, days_to_consider)
    print("Execution time:", time.time() - start_time, "seconds")
