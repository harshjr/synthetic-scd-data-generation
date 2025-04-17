import os
import subprocess
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from datetime import datetime, timedelta

def main():
    """Main function to run the entire SCD synthetic data generation project."""
    start_time = time.time()
    
    # Create project directory structure
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/synthetic_data', exist_ok=True)
    os.makedirs('results/evaluation', exist_ok=True)
    
    print("Starting SCD Synthetic Data Generation Project...")
    
    # Step 1: Data Exploration and Analysis
    print("\nStep 1: Data Exploration and Analysis")
    explore_data()
    
    # Step 2: Statistical Modeling and Synthetic Data Generation
    print("\nStep 2: Statistical Modeling and Synthetic Data Generation")
    generate_synthetic_data()
    
    # Step 3: Evaluation of Synthetic Data Quality
    print("\nStep 3: Evaluation of Synthetic Data Quality")
    evaluate_synthetic_data()
    
    # Generate a simple report
    generate_report()
    
    elapsed_time = time.time() - start_time
    print(f"\nProject completed in {elapsed_time:.2f} seconds.")
    print("Results are available in the 'results' directory.")
    print("A project report has been generated: 'project_report.md'")

def explore_data():
    """Data Exploration and Analysis"""
    # Set the style for plots
    plt.style.use('seaborn-v0_8-whitegrid')  # Updated style name for newer matplotlib versions
    sns.set_palette("deep")
    plt.rcParams['figure.figsize'] = [12, 8]

    # Load the SCD patient data
    file_path = '/Users/harshkumar/Downloads/scd_patients_all_100403.csv'
    df = pd.read_csv(file_path, sep=';')

    # Display basic information
    print("Dataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nFirst 5 rows:")
    print(df.head())

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Convert date columns to datetime
    date_columns = ['birthDate', 'diagDate', 'deathDate']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Calculate age at diagnosis
    df['diagAge'] = (df['diagDate'] - df['birthDate']).dt.days / 365.25

    # Basic statistics
    print("\nBasic Statistics:")
    print(df.describe())

    # Distribution of categorical variables
    categorical_cols = ['state', 'sex', 'race']
    for col in categorical_cols:
        plt.figure()
        counts = df[col].value_counts()
        plt.bar(counts.index, counts.values)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=90)
        plt.tight_layout()
        # Sanitize column name for filename
        safe_col_name = col.replace(':', '_').replace('/', '_').replace('\\', '_')
        plt.savefig(f'results/plots/distribution_{safe_col_name}.png')

    # Distribution of numerical variables
    numerical_cols = ['age', 'CBC:g/dL', 'RC:%', 'diagAge']
    for col in numerical_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        # Sanitize column name for filename
        safe_col_name = col.replace(':', '_').replace('/', '_').replace('\\', '_')
        plt.savefig(f'results/plots/distribution_{safe_col_name}.png')

    # Correlation matrix for numerical variables
    plt.figure(figsize=(10, 8))
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('results/plots/correlation_matrix.png')

    # Age distribution by race
    plt.figure()
    sns.boxplot(x='race', y='age', data=df)
    plt.title('Age Distribution by Race')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/plots/age_by_race.png')

    # CBC and RC distribution by sex
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(x='sex', y='CBC:g/dL', data=df, ax=axes[0])
    axes[0].set_title('CBC Distribution by Sex')
    sns.boxplot(x='sex', y='RC:%', data=df, ax=axes[1])
    axes[1].set_title('RC% Distribution by Sex')
    plt.tight_layout()
    plt.savefig('results/plots/blood_params_by_sex.png')

    # Save the processed data
    df.to_csv('results/processed_scd_data.csv', index=False)

    print("Data exploration completed. Plots saved in 'results/plots' directory.")
    
    return df

def generate_synthetic_data():
    """Statistical Modeling and Synthetic Data Generation"""
    # Load the processed data
    df = pd.read_csv('results/processed_scd_data.csv')

    # Convert date columns back to datetime
    date_columns = ['birthDate', 'diagDate', 'deathDate']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Define features for modeling
    categorical_features = ['state', 'sex', 'race']
    numerical_features = ['age', 'CBC:g/dL', 'RC:%', 'diagAge']

    # Generate synthetic data using different methods
    n_synthetic = 1000  # Number of synthetic samples to generate

    # 1. Method 1: Independent sampling for each column
    synthetic_data_method1 = pd.DataFrame()

    # Generate synthetic categorical data
    for col in categorical_features:
        if col in df.columns:
            synthetic_data_method1[col] = generate_categorical_synthetic_data(df, col, n_synthetic)

    # Generate synthetic numerical data
    for col in numerical_features:
        if col in df.columns:
            synthetic_data_method1[col] = generate_parametric_synthetic_data(df, col, n_synthetic)

    # Generate synthetic dates
    synthetic_data_method1['birthDate'] = generate_synthetic_dates(1930, 2023, n_synthetic)
    synthetic_data_method1['diagDate'] = [birth_date + timedelta(days=int(age*365.25)) 
                                         for birth_date, age in zip(synthetic_data_method1['birthDate'], 
                                                                   synthetic_data_method1['diagAge'])]

    # Generate synthetic IDs
    synthetic_data_method1['idx'] = np.random.randint(1, 100000, n_synthetic)

    # Generate synthetic zip codes
    synthetic_data_method1['zipCode'] = np.random.randint(10000, 99999, n_synthetic)

    # 2. Method 2: Multivariate normal for numerical features
    synthetic_numerical_method2 = generate_multivariate_synthetic_data(df, numerical_features, n_synthetic)

    synthetic_data_method2 = pd.DataFrame()
    # Copy categorical and date data from method 1
    for col in categorical_features + ['birthDate', 'diagDate', 'idx', 'zipCode']:
        if col in synthetic_data_method1.columns:
            synthetic_data_method2[col] = synthetic_data_method1[col].copy()

    # Add multivariate numerical data
    for col in numerical_features:
        if col in synthetic_numerical_method2.columns:
            synthetic_data_method2[col] = synthetic_numerical_method2[col]

    # 3. Method 3: Gaussian Mixture Model for numerical features
    synthetic_numerical_method3 = generate_gmm_synthetic_data(df, numerical_features, n_synthetic)

    synthetic_data_method3 = pd.DataFrame()
    # Copy categorical and date data from method 1
    for col in categorical_features + ['birthDate', 'diagDate', 'idx', 'zipCode']:
        if col in synthetic_data_method1.columns:
            synthetic_data_method3[col] = synthetic_data_method1[col].copy()

    # Add GMM numerical data
    for col in numerical_features:
        if col in synthetic_numerical_method3.columns:
            synthetic_data_method3[col] = synthetic_numerical_method3[col]

    # Save synthetic datasets
    synthetic_data_method1.to_csv('results/synthetic_data/synthetic_parametric.csv', index=False)
    synthetic_data_method2.to_csv('results/synthetic_data/synthetic_multivariate.csv', index=False)
    synthetic_data_method3.to_csv('results/synthetic_data/synthetic_gmm.csv', index=False)

    print("Synthetic data generation completed. Files saved in 'results/synthetic_data' directory.")

def generate_categorical_synthetic_data(data, column, n_samples):
    """Generate synthetic categorical data"""
    value_counts = data[column].value_counts(normalize=True)
    synthetic_values = np.random.choice(value_counts.index, size=n_samples, p=value_counts.values)
    return synthetic_values

def generate_parametric_synthetic_data(data, column, n_samples):
    """Generate synthetic numerical data using parametric method"""
    # Fit normal distribution
    mu, std = stats.norm.fit(data[column].dropna())
    
    # Generate synthetic data
    synthetic_values = np.random.normal(mu, std, n_samples)
    
    return synthetic_values

def generate_multivariate_synthetic_data(data, columns, n_samples):
    """Generate synthetic data using multivariate normal distribution"""
    # Select only rows with no missing values in the specified columns
    complete_data = data[columns].dropna()
    
    # Calculate mean vector and covariance matrix
    mean_vector = complete_data.mean().values
    cov_matrix = complete_data.cov().values
    
    # Generate synthetic data
    synthetic_values = np.random.multivariate_normal(mean_vector, cov_matrix, n_samples)
    
    # Convert to DataFrame with original column names
    synthetic_df = pd.DataFrame(synthetic_values, columns=columns)
    
    return synthetic_df

def generate_gmm_synthetic_data(data, columns, n_samples, n_components=3):
    """Generate synthetic data using Gaussian Mixture Model"""
    # Select only rows with no missing values in the specified columns
    complete_data = data[columns].dropna()
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(complete_data)
    
    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(scaled_data)
    
    # Generate synthetic data
    synthetic_scaled_values, _ = gmm.sample(n_samples)
    
    # Inverse transform to get back to original scale
    synthetic_values = scaler.inverse_transform(synthetic_scaled_values)
    
    # Convert to DataFrame with original column names
    synthetic_df = pd.DataFrame(synthetic_values, columns=columns)
    
    return synthetic_df

def generate_synthetic_dates(start_year, end_year, n_samples):
    """Generate synthetic dates"""
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    days_range = (end_date - start_date).days
    
    random_days = np.random.randint(0, days_range, n_samples)
    # Convert numpy.int64 to Python int
    synthetic_dates = [start_date + timedelta(days=int(days)) for days in random_days]
    
    return synthetic_dates

def evaluate_synthetic_data():
    """Evaluation of Synthetic Data Quality"""
    # Load the original and synthetic datasets
    original_data = pd.read_csv('results/processed_scd_data.csv')
    synthetic_parametric = pd.read_csv('results/synthetic_data/synthetic_parametric.csv')
    synthetic_multivariate = pd.read_csv('results/synthetic_data/synthetic_multivariate.csv')
    synthetic_gmm = pd.read_csv('results/synthetic_data/synthetic_gmm.csv')

    # Convert date columns to datetime
    date_columns = ['birthDate', 'diagDate', 'deathDate']
    for col in date_columns:
        if col in original_data.columns:
            original_data[col] = pd.to_datetime(original_data[col], errors='coerce')
        if col in synthetic_parametric.columns:
            synthetic_parametric[col] = pd.to_datetime(synthetic_parametric[col], errors='coerce')
        if col in synthetic_multivariate.columns:
            synthetic_multivariate[col] = pd.to_datetime(synthetic_multivariate[col], errors='coerce')
        if col in synthetic_gmm.columns:
            synthetic_gmm[col] = pd.to_datetime(synthetic_gmm[col], errors='coerce')

    # Define features for evaluation
    categorical_features = ['state', 'sex', 'race']
    numerical_features = ['age', 'CBC:g/dL', 'RC:%', 'diagAge']

    # Evaluate each synthetic dataset
    methods = {
        'Parametric': synthetic_parametric,
        'Multivariate': synthetic_multivariate,
        'GMM': synthetic_gmm
    }

    evaluation_results = {}

    for method_name, synthetic_data in methods.items():
        print(f"\nEvaluating {method_name} method:")
        
        method_results = {
            'categorical': {},
            'numerical': {},
            'correlation': None
        }
        
        # Evaluate categorical distributions
        for col in categorical_features:
            if col in original_data.columns and col in synthetic_data.columns:
                js_div, orig_dist, synth_dist = evaluate_categorical_distribution(original_data, synthetic_data, col)
                method_results['categorical'][col] = {
                    'js_divergence': js_div,
                    'original_distribution': orig_dist,
                    'synthetic_distribution': synth_dist
                }
                print(f"  {col} - Jensen-Shannon Divergence: {js_div:.4f}")
                
                # Plot comparison
                plt.figure(figsize=(12, 6))
                width = 0.35
                x = np.arange(len(orig_dist.index))
                plt.bar(x - width/2, orig_dist.values, width, label='Original')
                plt.bar(x + width/2, synth_dist.values, width, label='Synthetic')
                plt.xlabel('Categories')
                plt.ylabel('Frequency')
                plt.title(f'Distribution of {col} - {method_name} Method')
                plt.xticks(x, orig_dist.index, rotation=45)
                plt.legend()
                plt.tight_layout()
                # Sanitize column name for filename
                safe_col_name = col.replace(':', '_').replace('/', '_').replace('\\', '_')
                plt.savefig(f'results/evaluation/{method_name.lower()}_{safe_col_name}_comparison.png')
        
        # Evaluate numerical distributions
        for col in numerical_features:
            if col in original_data.columns and col in synthetic_data.columns:
                ks_stat, ks_pvalue, orig_stats, synth_stats = evaluate_numerical_distribution(original_data, synthetic_data, col)
                method_results['numerical'][col] = {
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    'original_stats': orig_stats,
                    'synthetic_stats': synth_stats
                }
                print(f"  {col} - KS Statistic: {ks_stat:.4f}, p-value: {ks_pvalue:.4f}")
                
                # Plot comparison
                plt.figure(figsize=(12, 6))
                sns.kdeplot(original_data[col].dropna(), label='Original', fill=True, alpha=0.3)
                sns.kdeplot(synthetic_data[col].dropna(), label='Synthetic', fill=True, alpha=0.3)
                plt.title(f'Distribution of {col} - {method_name} Method')
                plt.xlabel(col)
                plt.ylabel('Density')
                plt.legend()
                plt.tight_layout()
                # Sanitize column name for filename
                safe_col_name = col.replace(':', '_').replace('/', '_').replace('\\', '_')
                plt.savefig(f'results/evaluation/{method_name.lower()}_{safe_col_name}_comparison.png')
        
        # Evaluate correlation preservation
        frob_norm, orig_corr, synth_corr = evaluate_correlation_preservation(original_data, synthetic_data, numerical_features)
        method_results['correlation'] = {
            'frobenius_norm': frob_norm,
            'original_correlation': orig_corr,
            'synthetic_correlation': synth_corr
        }
        print(f"  Correlation Preservation - Frobenius Norm: {frob_norm:.4f}")
        
        # Plot correlation matrices
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        sns.heatmap(orig_corr, annot=True, cmap='coolwarm', ax=axes[0], vmin=-1, vmax=1)
        axes[0].set_title('Original Data Correlation')
        sns.heatmap(synth_corr, annot=True, cmap='coolwarm', ax=axes[1], vmin=-1, vmax=1)
        axes[1].set_title(f'Synthetic Data Correlation ({method_name})')
        plt.tight_layout()
        plt.savefig(f'results/evaluation/{method_name.lower()}_correlation_comparison.png')
        
        # Visualize original and synthetic data using t-SNE
        combined_numerical = pd.concat([
            original_data[numerical_features].sample(min(1000, len(original_data))).assign(type='Original'),
            synthetic_data[numerical_features].sample(min(1000, len(synthetic_data))).assign(type='Synthetic')
        ])
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(combined_numerical[numerical_features])
        
        # Plot t-SNE results
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                             c=[0 if t == 'Original' else 1 for t in combined_numerical['type']], 
                             alpha=0.6, cmap='coolwarm')
        plt.colorbar(scatter, ticks=[0, 1], label='Data Type')
        plt.clim(-0.5, 1.5)
        plt.title(f't-SNE Visualization - {method_name} Method')
        plt.tight_layout()
        plt.savefig(f'results/evaluation/{method_name.lower()}_tsne_visualization.png')
        
        evaluation_results[method_name] = method_results

    # Save evaluation results
    with open('results/evaluation/evaluation_summary.txt', 'w') as f:
        f.write("Synthetic Data Evaluation Summary\n")
        f.write("===============================\n\n")
        
        for method_name, results in evaluation_results.items():
            f.write(f"{method_name} Method:\n")
            f.write("-----------------\n")
            
            f.write("Categorical Variables:\n")
            for col, metrics in results['categorical'].items():
                f.write(f"  {col} - Jensen-Shannon Divergence: {metrics['js_divergence']:.4f}\n")
            
            f.write("\nNumerical Variables:\n")
            for col, metrics in results['numerical'].items():
                f.write(f"  {col} - KS Statistic: {metrics['ks_statistic']:.4f}, p-value: {metrics['ks_pvalue']:.4f}\n")
            
            f.write(f"\nCorrelation Preservation - Frobenius Norm: {results['correlation']['frobenius_norm']:.4f}\n\n")

    print("\nEvaluation completed. Results saved in 'results/evaluation' directory.")

def evaluate_categorical_distribution(original, synthetic, column):
    """Evaluate categorical distributions"""
    orig_counts = original[column].value_counts(normalize=True)
    synth_counts = synthetic[column].value_counts(normalize=True)
    
    # Align the indices
    all_categories = sorted(set(orig_counts.index) | set(synth_counts.index))
    orig_aligned = pd.Series([orig_counts.get(cat, 0) for cat in all_categories], index=all_categories)
    synth_aligned = pd.Series([synth_counts.get(cat, 0) for cat in all_categories], index=all_categories)
    
    # Calculate Jensen-Shannon divergence
    m = 0.5 * (orig_aligned + synth_aligned)
    js_divergence = 0.5 * (stats.entropy(orig_aligned, m) + stats.entropy(synth_aligned, m))
    
    return js_divergence, orig_aligned, synth_aligned

def evaluate_numerical_distribution(original, synthetic, column):
    """Evaluate numerical distributions"""
    # Calculate basic statistics
    orig_stats = original[column].describe()
    synth_stats = synthetic[column].describe()
    
    # Calculate Kolmogorov-Smirnov statistic
    ks_stat, ks_pvalue = stats.ks_2samp(original[column].dropna(), synthetic[column].dropna())
    
    return ks_stat, ks_pvalue, orig_stats, synth_stats

def evaluate_correlation_preservation(original, synthetic, columns):
    """Evaluate correlation preservation"""
    orig_corr = original[columns].corr()
    synth_corr = synthetic[columns].corr()
    
    # Calculate Frobenius norm of the difference
    frob_norm = np.linalg.norm(orig_corr - synth_corr, 'fro')
    
    return frob_norm, orig_corr, synth_corr

def generate_report():
    """Generate a simple project report"""
    with open('project_report.md', 'w') as f:
        f.write("# Synthetic Data Generation for SCD Patients\n\n")
        f.write("## Project Overview\n")
        f.write("This project generates synthetic data for Sickle Cell Disease (SCD) patients using statistical methods.\n\n")
        
        f.write("## Methods Used\n")
        f.write("1. **Parametric Method**: Independent sampling from fitted distributions for each variable\n")
        f.write("2. **Multivariate Normal Method**: Preserves correlations between numerical variables\n")
        f.write("3. **Gaussian Mixture Model**: Captures complex distributions and relationships\n\n")
        
        f.write("## Results\n")
        f.write("The evaluation results can be found in the 'results/evaluation' directory.\n")
        f.write("Synthetic datasets are available in the 'results/synthetic_data' directory.\n\n")
        
        f.write("## Conclusion\n")
        f.write("Based on the evaluation metrics, the most suitable method for generating synthetic SCD patient data is [to be determined based on evaluation results].\n")

if __name__ == "__main__":
    main()