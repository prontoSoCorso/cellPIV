import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy import stats
import plotly.express as px
import statsmodels.api as sm
from datetime import datetime
import logging
import webbrowser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_blastocyst_proportions(input_csv_path, output_dir):
    """Main analysis function with enhanced features"""
    try:
        # 1. Load and validate data
        df = load_and_validate_data(input_csv_path)
        
        # 2. Process patient data
        patient_data = process_patient_data(df)
        
        # 3. Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 4. Generate visualizations
        generate_scatter_plots(patient_data, output_dir)
        generate_boxplots(patient_data, output_dir)
        generate_age_distribution(patient_data, output_dir)
        html_path = generate_interactive_plot(patient_data, output_dir)
        
        logging.info(f"Interactive plot available at: {html_path}")
        
        # 5. Statistical analysis
        perform_statistical_analysis(patient_data, output_dir)
        
        # 6. Save metadata
        save_analysis_metadata(patient_data, output_dir)
        
        logging.info("Analysis completed successfully")

    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise

def load_and_validate_data(file_path):
    """Load and validate input data"""
    logging.info("Loading and validating data")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Validate required columns
    required_columns = ['patient_id', 'maternal age', 'BLASTO NY']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean data
    df['maternal age'] = pd.to_numeric(df['maternal age'], errors='coerce')
    df = df.dropna(subset=['maternal age', 'BLASTO NY'])
    
    return df

def process_patient_data(df):
    """Process patient-level data"""
    logging.info("Processing patient data")
    
    # Aggregate patient data
    patient_data = df.groupby('patient_id').agg(
        age=('maternal age', 'first'),
        blast_count=('BLASTO NY', 'sum'),
        total_embryos=('BLASTO NY', 'count')
    ).reset_index()
    
    # Calculate proportions
    patient_data['proportion_blast'] = (
        patient_data['blast_count'] / patient_data['total_embryos']
    )
    
    # Data quality checks
    if (patient_data['total_embryos'] == 0).any():
        raise ValueError("Found patients with 0 embryos - check data quality")
    
    if (patient_data['age'] < 18).any() or (patient_data['age'] > 50).any():
        logging.warning("Unusual age values detected (<18 or >50)")
    
    return patient_data

def generate_scatter_plots(data, output_dir):
    """Generate scatter plots with legend fix"""
    logging.info("Generating scatter plots")
    
    plt.figure(figsize=(12, 7))
    ax = sns.scatterplot(
        data=data,
        x='age',
        y='proportion_blast',
        size='total_embryos',
        sizes=(20, 200),
        alpha=0.6,
        hue='total_embryos',
        palette='viridis',
        legend='full'
    )
    
    sns.regplot(
        data=data,
        x='age',
        y='proportion_blast',
        scatter=False,
        color='red',
        ci=95,
        label='Trend (95% CI)'
    )
    
    # Correlation annotation
    corr, pval = stats.pearsonr(data['age'], data['proportion_blast'])
    plt.text(0.05, 0.95, 
             f'Pearson r = {corr:.2f}\np = {pval:.3f}', 
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title('Blastocyst Proportions by Maternal Age', fontsize=14)
    plt.xlabel('Maternal Age (years)', fontsize=12)
    plt.ylabel('Proportion of Blastocysts', fontsize=12)
    
    # Improved legend
    legend = plt.legend(
        title='Embryo Count',
        bbox_to_anchor=(1.15, 1),
        loc='upper left',
        frameon=False,
        ncol=2,
        markerscale=0.8
    )
    
    # Adjust layout
    plt.subplots_adjust(right=0.78)
    
    plt.savefig(os.path.join(output_dir, 'blast_proportion_scatter.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def generate_boxplots(data, output_dir):
    """Generate enhanced boxplots"""
    logging.info("Generating boxplots")
    
    # Dynamic age binning
    data['age_group'] = pd.qcut(data['age'], q=4, precision=0)
    
    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=data,
        x='age_group',
        y='proportion_blast',
        color='skyblue',
        showmeans=True,
        meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black'}
    )
    sns.stripplot(
        data=data,
        x='age_group',
        y='proportion_blast',
        color='black',
        alpha=0.4,
        jitter=0.2
    )
    
    # Add statistical annotations
    add_statistical_annotations(data)
    
    plt.title('Blastocyst Proportions by Age Group', fontsize=14)
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel('Proportion of Blastocysts', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'blast_proportion_boxplot.png'), dpi=300)
    plt.close()

def add_statistical_annotations(data):
    """Add statistical test results to boxplot"""
    groups = sorted(data['age_group'].unique())
    pairs = [(groups[i], groups[j]) for i in range(len(groups)) for j in range(i+1, len(groups))]
    
    # Perform ANOVA
    group_data = [data[data['age_group'] == grp]['proportion_blast'] for grp in groups]
    f_val, p_val = stats.f_oneway(*group_data)
    
    plt.text(0.95, 0.95, 
             f'ANOVA: F = {f_val:.2f}\np = {p_val:.3f}', 
             transform=plt.gca().transAxes,
             ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.8))

def generate_age_distribution(data, output_dir):
    """Generate age distribution plot"""
    logging.info("Generating age distribution plot")
    
    plt.figure(figsize=(10, 5))
    sns.histplot(data['age'], bins=20, kde=True, color='teal')
    plt.title('Maternal Age Distribution', fontsize=14)
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Number of Patients', fontsize=12)
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'), dpi=300)
    plt.close()

def generate_interactive_plot(data, output_dir):
    """Generate interactive plot with browser opening"""
    logging.info("Generating interactive plot")
    
    try:
        fig = px.scatter(
            data,
            x='age',
            y='proportion_blast',
            size='total_embryos',
            color='total_embryos',
            hover_data=['patient_id', 'age', 'proportion_blast'],
            trendline="lowess",
            title="Blastocyst Proportion Analysis",
            labels={
                'age': 'Maternal Age',
                'proportion_blast': 'Blastocyst Proportion'
            }
        )
        
        html_path = os.path.join(output_dir, "interactive_plot.html")
        fig.write_html(html_path)
        
        # Try to open in browser
        try:
            webbrowser.open(f'file://{os.path.abspath(html_path)}')
        except Exception as e:
            logging.info(f"Open manually: {os.path.abspath(html_path)}")
            
        return html_path
        
    except Exception as e:
        logging.warning(f"Could not generate interactive plot: {str(e)}")
        return None

def perform_statistical_analysis(data, output_dir):
    """Perform advanced statistical modeling"""
    logging.info("Performing statistical analysis")
    
    # Logistic regression
    X = sm.add_constant(data['age'])
    y = data[['blast_count', 'total_embryos']]
    
    model = sm.GLM(y, X, family=sm.families.Binomial())
    results = model.fit()
    
    # Save results
    with open(os.path.join(output_dir, 'statistical_analysis.txt'), 'w') as f:
        f.write("Logistic Regression Results:\n")
        f.write(results.summary().as_text())
        f.write("\n\nKey Statistics:\n")
        f.write(f"- Pearson correlation: {stats.pearsonr(data['age'], data['proportion_blast'])}\n")
        f.write(f"- Average proportion: {data['proportion_blast'].mean():.2f}\n")
        f.write(f"- Median age: {data['age'].median()} years\n")

def save_analysis_metadata(data, output_dir):
    """Save analysis metadata"""
    metadata = {
        'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_patients': len(data),
        'age_range': f"{data['age'].min()} - {data['age'].max()}",
        'total_embryos': data['total_embryos'].sum(),
        'mean_proportion': data['proportion_blast'].mean()
    }
    
    with open(os.path.join(output_dir, 'analysis_metadata.txt'), 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

if __name__ == '__main__':
    import sys
    import os

    # Configure paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    while not os.path.basename(parent_dir) == "cellPIV":
        parent_dir = os.path.dirname(parent_dir)
    sys.path.append(parent_dir)
    
    from config import Config_00_preprocessing as conf
    
    try:
        plot_blastocyst_proportions(
            input_csv_path=conf.path_addedID_csv,
            output_dir=os.path.join(current_dir, "blast_proportions_analysis")
        )
    except Exception as e:
        logging.error(f"Critical error in main execution: {str(e)}")
        sys.exit(1)