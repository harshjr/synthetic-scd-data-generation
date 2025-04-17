import dash
from dash import dcc, html, callback, Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Load data
def load_data():
    try:
        # Load original data
        print("Attempting to load original data...")
        if os.path.exists('results/processed_scd_data.csv'):
            original_data = pd.read_csv('results/processed_scd_data.csv')
            print(f"Successfully loaded original data: {len(original_data)} records")
        else:
            print("Warning: Original data file not found at 'results/processed_scd_data.csv'")
            original_data = pd.DataFrame()
        
        # Load synthetic data
        synthetic_data = {}
        synthetic_files = {
            'Parametric': 'results/synthetic_data/synthetic_parametric.csv',
            'Multivariate': 'results/synthetic_data/synthetic_multivariate.csv',
            'GMM': 'results/synthetic_data/synthetic_gmm.csv'
        }
        
        print("Attempting to load synthetic data...")
        for method, file_path in synthetic_files.items():
            if os.path.exists(file_path):
                synthetic_data[method] = pd.read_csv(file_path)
                print(f"Loaded {method} data: {len(synthetic_data[method])} records")
            else:
                print(f"Warning: {file_path} not found")
        
        # If no data was loaded, create some sample data for testing
        if original_data.empty and not synthetic_data:
            print("No data found. Creating sample data for testing...")
            # Create sample data
            original_data = pd.DataFrame({
                'age': np.random.normal(50, 15, 100),
                'height': np.random.normal(170, 10, 100),
                'weight': np.random.normal(70, 15, 100),
                'gender': np.random.choice(['Male', 'Female'], 100),
                'blood_type': np.random.choice(['A', 'B', 'AB', 'O'], 100)
            })
            
            synthetic_data['Sample'] = pd.DataFrame({
                'age': np.random.normal(48, 16, 100),
                'height': np.random.normal(168, 12, 100),
                'weight': np.random.normal(72, 14, 100),
                'gender': np.random.choice(['Male', 'Female'], 100),
                'blood_type': np.random.choice(['A', 'B', 'AB', 'O'], 100)
            })
            
            print("Created sample data for testing")
        
        return original_data, synthetic_data
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), {}

# Load the data
original_data, synthetic_data = load_data()

# Get list of methods
methods = list(synthetic_data.keys())
all_methods = ['Original'] + methods

# Get numerical and categorical columns
if not original_data.empty:
    numerical_cols = original_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = original_data.select_dtypes(include=['object', 'category']).columns.tolist()
    # Remove date columns from categorical
    categorical_cols = [col for col in categorical_cols if 'date' not in col.lower()]
else:
    numerical_cols = []
    categorical_cols = []

# Define colors for consistent styling
colors = {
    'background': '#f9f9f9',
    'text': '#333333',
    'primary': '#4CAF50',
    'secondary': '#2196F3',
    'accent': '#FF9800',
    'panel': '#ffffff'
}

# Custom CSS for better styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Roboto', sans-serif;
                margin: 0;
                background-color: ''' + colors['background'] + ''';
                color: ''' + colors['text'] + ''';
            }
            .dashboard-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .dashboard-header {
                background-color: ''' + colors['primary'] + ''';
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .dashboard-panel {
                background-color: ''' + colors['panel'] + ''';
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .panel-header {
                color: ''' + colors['primary'] + ''';
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
                margin-top: 0;
            }
            .control-group {
                margin-bottom: 15px;
            }
            .control-label {
                font-weight: bold;
                margin-bottom: 5px;
                display: block;
            }
            table {
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: ''' + colors['secondary'] + ''';
                color: white;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Improved layout with better UI components and additional evaluation graphs
app.layout = html.Div(className='dashboard-container', children=[
    html.Div(className='dashboard-header', children=[
        html.H1("SCD Synthetic Data Dashboard", style={'textAlign': 'center', 'marginBottom': '0'}),
        html.P("Interactive visualization of original and synthetic Sickle Cell Disease data", 
               style={'textAlign': 'center', 'fontSize': '1.2em', 'marginTop': '5px'})
    ]),
    
    html.Div(className='dashboard-panel', children=[
        html.H3("Data Overview", className='panel-header'),
        html.Div([
            html.P(f"Original Data: {len(original_data)} records" if not original_data.empty else "Original data not loaded"),
            html.Div([
                html.P(f"{method} Data: {len(data)} records") 
                for method, data in synthetic_data.items()
            ])
        ])
    ]),
    
    # Distribution Comparison and t-SNE Visualization
    html.Div(className='row', style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '0 -10px'}, children=[
        html.Div(className='dashboard-panel', style={'flex': '1', 'minWidth': '45%', 'margin': '0 10px'}, children=[
            html.H3("Data Distribution Comparison", className='panel-header'),
            html.Div(className='control-group', children=[
                html.Label("Select Variable Type:", className='control-label'),
                dcc.RadioItems(
                    id='variable-type',
                    options=[
                        {'label': 'Numerical', 'value': 'numerical'},
                        {'label': 'Categorical', 'value': 'categorical'}
                    ],
                    value='numerical',
                    labelStyle={'display': 'inline-block', 'marginRight': '15px'},
                    style={'marginBottom': '10px'}
                ),
                html.Label("Select Variable:", className='control-label'),
                dcc.Dropdown(
                    id='variable-dropdown',
                    options=[{'label': col, 'value': col} for col in numerical_cols],
                    value=numerical_cols[0] if numerical_cols else None,
                    style={'marginBottom': '15px'}
                ),
            ]),
            dcc.Graph(id='distribution-plot', style={'height': '400px'})
        ]),
        
        html.Div(className='dashboard-panel', style={'flex': '1', 'minWidth': '45%', 'margin': '0 10px'}, children=[
            html.H3("t-SNE Visualization", className='panel-header'),
            html.Div(className='control-group', children=[
                html.Label("Select Methods to Compare:", className='control-label'),
                dcc.Checklist(
                    id='tsne-methods',
                    options=[{'label': method, 'value': method} for method in all_methods],
                    value=['Original', methods[0]] if methods else ['Original'],
                    labelStyle={'display': 'block', 'marginBottom': '5px'},
                    style={'marginBottom': '15px'}
                ),
            ]),
            dcc.Graph(id='tsne-plot', style={'height': '400px'})
        ]),
    ]),
    
    # Box Plot Comparison
    html.Div(className='dashboard-panel', children=[
        html.H3("Box Plot Comparison", className='panel-header'),
        html.Div(className='control-group', children=[
            html.Label("Select Variable:", className='control-label'),
            dcc.Dropdown(
                id='boxplot-variable',
                options=[{'label': col, 'value': col} for col in numerical_cols],
                value=numerical_cols[0] if numerical_cols else None,
                style={'marginBottom': '15px', 'maxWidth': '300px'}
            ),
        ]),
        dcc.Graph(id='boxplot-comparison', style={'height': '400px'})
    ]),
    
    # Correlation Matrix
    html.Div(className='dashboard-panel', children=[
        html.H3("Correlation Matrix Comparison", className='panel-header'),
        html.Div(className='control-group', children=[
            html.Label("Select Method:", className='control-label'),
            dcc.Dropdown(
                id='correlation-method',
                options=[{'label': method, 'value': method} for method in all_methods],
                value='Original',
                style={'marginBottom': '15px', 'maxWidth': '300px'}
            ),
        ]),
        dcc.Graph(id='correlation-plot', style={'height': '500px'})
    ]),
    
    # Pair Plot
    html.Div(className='dashboard-panel', children=[
        html.H3("Pair Plot", className='panel-header'),
        html.Div(className='control-group', children=[
            html.Label("Select Method:", className='control-label'),
            dcc.Dropdown(
                id='pairplot-method',
                options=[{'label': method, 'value': method} for method in all_methods],
                value='Original',
                style={'marginBottom': '10px', 'maxWidth': '300px'}
            ),
            html.Label("Select Variables (max 4):", className='control-label'),
            dcc.Dropdown(
                id='pairplot-variables',
                options=[{'label': col, 'value': col} for col in numerical_cols],
                value=numerical_cols[:min(4, len(numerical_cols))] if numerical_cols else [],
                multi=True,
                style={'marginBottom': '15px'}
            ),
        ]),
        dcc.Graph(id='pair-plot', style={'height': '600px'})
    ]),
    
    # Violin Plot Comparison
    html.Div(className='dashboard-panel', children=[
        html.H3("Violin Plot Comparison", className='panel-header'),
        html.Div(className='control-group', children=[
            html.Label("Select Variable:", className='control-label'),
            dcc.Dropdown(
                id='violin-variable',
                options=[{'label': col, 'value': col} for col in numerical_cols],
                value=numerical_cols[0] if numerical_cols else None,
                style={'marginBottom': '15px', 'maxWidth': '300px'}
            ),
        ]),
        dcc.Graph(id='violin-comparison', style={'height': '400px'})
    ]),
    
    # Cumulative Distribution Function (CDF) Plot
    html.Div(className='dashboard-panel', children=[
        html.H3("Cumulative Distribution Function (CDF)", className='panel-header'),
        html.Div(className='control-group', children=[
            html.Label("Select Variable:", className='control-label'),
            dcc.Dropdown(
                id='cdf-variable',
                options=[{'label': col, 'value': col} for col in numerical_cols],
                value=numerical_cols[0] if numerical_cols else None,
                style={'marginBottom': '15px', 'maxWidth': '300px'}
            ),
        ]),
        dcc.Graph(id='cdf-plot', style={'height': '400px'})
    ]),
    
    # Categorical Data Comparison
    html.Div(className='dashboard-panel', children=[
        html.H3("Categorical Data Comparison", className='panel-header'),
        html.Div(className='control-group', children=[
            html.Label("Select Categorical Variable:", className='control-label'),
            dcc.Dropdown(
                id='categorical-variable',
                options=[{'label': col, 'value': col} for col in categorical_cols],
                value=categorical_cols[0] if categorical_cols else None,
                style={'marginBottom': '15px', 'maxWidth': '300px'}
            ),
        ]),
        dcc.Graph(id='categorical-comparison', style={'height': '400px'})
    ]),
    
    # Data Statistics and Quality Metrics
    html.Div(className='row', style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '0 -10px'}, children=[
        html.Div(className='dashboard-panel', style={'flex': '1', 'minWidth': '45%', 'margin': '0 10px'}, children=[
            html.H3("Data Statistics", className='panel-header'),
            html.Div(className='control-group', children=[
                html.Label("Select Method:", className='control-label'),
                dcc.Dropdown(
                    id='stats-method',
                    options=[{'label': method, 'value': method} for method in all_methods],
                    value='Original',
                    style={'marginBottom': '10px'}
                ),
                html.Label("Select Variable:", className='control-label'),
                dcc.Dropdown(
                    id='stats-variable',
                    options=[{'label': col, 'value': col} for col in numerical_cols],
                    value=numerical_cols[0] if numerical_cols else None,
                    style={'marginBottom': '15px'}
                ),
            ]),
            html.Div(id='stats-table', style={'overflowX': 'auto'})
        ]),
        
        html.Div(className='dashboard-panel', style={'flex': '1', 'minWidth': '45%', 'margin': '0 10px'}, children=[
            html.H3("Data Quality Metrics", className='panel-header'),
            html.Div(id='quality-metrics', children=[
                html.P("Select methods in the t-SNE visualization to see quality metrics between original and synthetic data."),
                html.Div(id='quality-content')
            ])
        ]),
    ]),
    
    # Variable Importance Plot
    html.Div(className='dashboard-panel', children=[
        html.H3("Variable Importance", className='panel-header'),
        html.Div(className='control-group', children=[
            html.Label("Select Method:", className='control-label'),
            dcc.Dropdown(
                id='importance-method',
                options=[{'label': method, 'value': method} for method in all_methods],
                value='Original',
                style={'marginBottom': '15px', 'maxWidth': '300px'}
            ),
        ]),
        dcc.Graph(id='variable-importance', style={'height': '500px'})
    ]),
    
    # Missing Value Analysis
    html.Div(className='dashboard-panel', children=[
        html.H3("Missing Value Analysis", className='panel-header'),
        html.Div(className='control-group', children=[
            html.Label("Select Method:", className='control-label'),
            dcc.Dropdown(
                id='missing-method',
                options=[{'label': method, 'value': method} for method in all_methods],
                value='Original',
                style={'marginBottom': '15px', 'maxWidth': '300px'}
            ),
        ]),
        dcc.Graph(id='missing-values-plot', style={'height': '400px'})
    ]),
    
    # Age Distribution by Method
    html.Div(className='dashboard-panel', children=[
        html.H3("Age Distribution by Method", className='panel-header'),
        dcc.Graph(id='age-distribution', style={'height': '400px'})
    ]),
    
    # Scatter Plot Matrix
    html.Div(className='dashboard-panel', children=[
        html.H3("Scatter Plot Comparison", className='panel-header'),
        html.Div(className='control-group', children=[
            html.Label("Select X Variable:", className='control-label'),
            dcc.Dropdown(
                id='scatter-x-variable',
                options=[{'label': col, 'value': col} for col in numerical_cols],
                value=numerical_cols[0] if len(numerical_cols) > 0 else None,
                style={'marginBottom': '15px', 'maxWidth': '300px'}
            ),
            html.Label("Select Y Variable:", className='control-label'),
            dcc.Dropdown(
                id='scatter-y-variable',
                options=[{'label': col, 'value': col} for col in numerical_cols],
                value=numerical_cols[1] if len(numerical_cols) > 1 else numerical_cols[0] if len(numerical_cols) > 0 else None,
                style={'marginBottom': '15px', 'maxWidth': '300px'}
            ),
        ]),
        dcc.Graph(id='scatter-comparison', style={'height': '500px'})
    ]),
    
    # Data Completeness
    html.Div(className='dashboard-panel', children=[
        html.H3("Data Completeness Comparison", className='panel-header'),
        dcc.Graph(id='completeness-comparison', style={'height': '500px'})
    ]),
])

# Callbacks for interactive components
@callback(
    Output('variable-dropdown', 'options'),
    Input('variable-type', 'value')
)
def update_dropdown_options(variable_type):
    if variable_type == 'numerical':
        return [{'label': col, 'value': col} for col in numerical_cols]
    else:
        return [{'label': col, 'value': col} for col in categorical_cols]

@callback(
    Output('variable-dropdown', 'value'),
    Input('variable-dropdown', 'options')
)
def update_dropdown_value(available_options):
    if available_options and len(available_options) > 0:
        return available_options[0]['value']
    return None

@callback(
    Output('distribution-plot', 'figure'),
    Input('variable-type', 'value'),
    Input('variable-dropdown', 'value')
)
def update_distribution_plot(variable_type, variable):
    if not variable or original_data.empty:
        return go.Figure().update_layout(
            title="No data available",
            xaxis_title="Value",
            yaxis_title="Count",
            template="plotly_white"
        )
    
    fig = go.Figure()
    
    try:
        if variable_type == 'numerical':
            # Add histogram for original data
            if variable in original_data.columns:
                fig.add_trace(go.Histogram(
                    x=original_data[variable].dropna(),
                    name='Original',
                    opacity=0.7,
                    nbinsx=30,
                    marker_color=colors['primary']
                ))
            
            # Add histograms for synthetic data
            for i, method in enumerate(methods):
                if method in synthetic_data and variable in synthetic_data[method].columns:
                    fig.add_trace(go.Histogram(
                        x=synthetic_data[method][variable].dropna(),
                        name=method,
                        opacity=0.7,
                        nbinsx=30,
                        marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                    ))
            
            fig.update_layout(
                barmode='overlay',
                title=f'Distribution of {variable}',
                xaxis_title=variable,
                yaxis_title='Count',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                template="plotly_white"
            )
        else:
            # For categorical variables, use bar charts
            if variable in original_data.columns:
                orig_counts = original_data[variable].value_counts(normalize=True).reset_index()
                orig_counts.columns = [variable, 'count']
                fig.add_trace(go.Bar(
                    x=orig_counts[variable],
                    y=orig_counts['count'],
                    name='Original',
                    marker_color=colors['primary']
                ))
            
            for i, method in enumerate(methods):
                if method in synthetic_data and variable in synthetic_data[method].columns:
                    syn_counts = synthetic_data[method][variable].value_counts(normalize=True).reset_index()
                    syn_counts.columns = [variable, 'count']
                    fig.add_trace(go.Bar(
                        x=syn_counts[variable],
                        y=syn_counts['count'],
                        name=method,
                        marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                    ))
            
            fig.update_layout(
                title=f'Distribution of {variable}',
                xaxis_title=variable,
                yaxis_title='Proportion',
                barmode='group',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                template="plotly_white"
            )
    except Exception as e:
        print(f"Error in distribution plot: {e}")
        fig = go.Figure().update_layout(
            title=f"Error generating plot: {str(e)}",
            template="plotly_white"
        )
    
    return fig

@callback(
    Output('tsne-plot', 'figure'),
    Input('tsne-methods', 'value')
)
def update_tsne_plot(selected_methods):
    if not selected_methods:
        return go.Figure().update_layout(
            title="Please select at least one method for t-SNE visualization",
            template="plotly_white"
        )
    
    # Debug information
    print(f"t-SNE selected methods: {selected_methods}")
    print(f"Original data empty: {original_data.empty}")
    print(f"Available synthetic methods: {list(synthetic_data.keys())}")
    
    # Check if we have any data to work with
    if original_data.empty and not synthetic_data:
        return go.Figure().update_layout(
            title="No data available. Please check file paths in console output.",
            template="plotly_white"
        )
    
    # Prepare data for t-SNE
    combined_data = []
    labels = []
    
    # Select only numerical columns for t-SNE
    if not original_data.empty:
        num_cols = [col for col in numerical_cols if col in original_data.columns]
        print(f"Numerical columns: {num_cols[:5]}{'...' if len(num_cols) > 5 else ''}")
    else:
        # If original data is empty but we have synthetic data, use columns from first synthetic dataset
        for method in synthetic_data:
            if not synthetic_data[method].empty:
                num_cols = synthetic_data[method].select_dtypes(include=['float64', 'int64']).columns.tolist()
                print(f"Using numerical columns from {method} data: {num_cols[:5]}{'...' if len(num_cols) > 5 else ''}")
                break
        else:
            num_cols = []
    
    if not num_cols:
        return go.Figure().update_layout(
            title="No numerical columns available for t-SNE visualization",
            template="plotly_white"
        )
    
    try:
        # Sample data to prevent memory issues (t-SNE can be memory intensive)
        sample_size = 100  # Reduced sample size for faster processing
        
        if 'Original' in selected_methods and not original_data.empty:
            # Make sure all columns exist in the data
            valid_cols = [col for col in num_cols if col in original_data.columns]
            if valid_cols:
                orig_data = original_data[valid_cols].copy()
                # Handle NaN values
                orig_data = orig_data.dropna()
                if len(orig_data) > sample_size:
                    orig_data = orig_data.sample(sample_size, random_state=42)
                if not orig_data.empty:
                    print(f"Adding {len(orig_data)} samples from Original data")
                    combined_data.append(orig_data)
                    labels.extend(['Original'] * len(orig_data))
        
        for method in selected_methods:
            if method != 'Original' and method in synthetic_data:
                # Make sure all columns exist in the data
                valid_cols = [col for col in num_cols if col in synthetic_data[method].columns]
                if valid_cols:
                    syn_data = synthetic_data[method][valid_cols].copy()
                    # Handle NaN values
                    syn_data = syn_data.dropna()
                    if len(syn_data) > sample_size:
                        syn_data = syn_data.sample(sample_size, random_state=42)
                    if not syn_data.empty:
                        print(f"Adding {len(syn_data)} samples from {method} data")
                        combined_data.append(syn_data)
                        labels.extend([method] * len(syn_data))
        
        if not combined_data:
            return go.Figure().update_layout(
                title="No valid data available for selected methods after filtering",
                template="plotly_white"
            )
        
        # Combine data
        print(f"Combining {len(combined_data)} datasets")
        combined_df = pd.concat(combined_data, axis=0)
        
        # Handle potential issues with the data
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.dropna()
        
        if combined_df.empty:
            return go.Figure().update_layout(
                title="No valid data after removing NaN/Inf values",
                template="plotly_white"
            )
        
        print(f"Final combined data shape: {combined_df.shape}")
        print(f"Number of labels: {len(labels)}")
        
        # Make sure labels match the dataframe length
        labels = labels[:len(combined_df)]
        
        # Scale the data
        print("Scaling data...")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(combined_df)
        
        # Apply t-SNE with appropriate perplexity
        print("Running t-SNE...")
        # Ensure perplexity is valid (must be less than n_samples - 1)
        perplexity = min(30, max(5, len(scaled_data) // 10))
        perplexity = min(perplexity, len(scaled_data) - 2)
        print(f"Using perplexity: {perplexity}")
        
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            n_iter=1000,
            verbose=1
        )
        tsne_results = tsne.fit_transform(scaled_data)
        
        # Create DataFrame for plotting
        print("Creating plot...")
        tsne_df = pd.DataFrame({
            'TSNE1': tsne_results[:, 0],
            'TSNE2': tsne_results[:, 1],
            'Method': labels
        })
        
        # Create plot
        fig = px.scatter(
            tsne_df, x='TSNE1', y='TSNE2', color='Method',
            title='t-SNE Visualization of Data Distribution',
            labels={'TSNE1': 't-SNE Component 1', 'TSNE2': 't-SNE Component 2'},
            template="plotly_white"
        )
        
        return fig
    
    except Exception as e:
        import traceback
        print(f"Error in t-SNE calculation: {e}")
        print(traceback.format_exc())
        return go.Figure().update_layout(
            title=f"Error in t-SNE calculation: {str(e)}",
            template="plotly_white"
        )

@callback(
    Output('correlation-plot', 'figure'),
    Input('correlation-method', 'value')
)
def update_correlation_plot(method):
    if method == 'Original':
        data = original_data
    else:
        data = synthetic_data.get(method, pd.DataFrame())
    
    if data.empty:
        return go.Figure().update_layout(
            title="No data available for correlation matrix",
            template="plotly_white"
        )
    
    try:
        # Calculate correlation matrix for numerical columns
        num_cols = [col for col in numerical_cols if col in data.columns]
        
        if not num_cols:
            return go.Figure().update_layout(
                title="No numerical columns available for correlation matrix",
                template="plotly_white"
            )
        
        # Use only numeric columns that don't have NaN values
        valid_cols = []
        for col in num_cols:
            if not data[col].isnull().any():
                valid_cols.append(col)
        
        if not valid_cols:
            return go.Figure().update_layout(
                title="No valid columns without NaN values for correlation matrix",
                template="plotly_white"
            )
        
        corr_matrix = data[valid_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(title='Correlation')
        ))
        
        fig.update_layout(
            title=f'Correlation Matrix for {method} Data',
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange='reversed'),
            template="plotly_white",
            height=600,
            width=800
        )
        
        return fig
    except Exception as e:
        print(f"Error in correlation plot: {e}")
        return go.Figure().update_layout(
            title=f"Error generating correlation matrix: {str(e)}",
            template="plotly_white"
        )

@callback(
    Output('stats-table', 'children'),
    Input('stats-method', 'value'),
    Input('stats-variable', 'value')
)
def update_stats_table(method, variable):
    if not variable:
        return html.Div("No variable selected")
    
    if method == 'Original':
        data = original_data
    else:
        data = synthetic_data.get(method, pd.DataFrame())
    
    if data.empty or variable not in data.columns:
        return html.Div("Data not available")
    
    try:
        # Calculate statistics
        stats = data[variable].describe()
        
        # Create table with better styling
        table_rows = []
        for stat, value in stats.items():
            table_rows.append(html.Tr([
                html.Td(stat),
                html.Td(f"{value:.4f}" if isinstance(value, float) else str(value))
            ]))
        
        return html.Table([
            html.Thead(html.Tr([html.Th("Statistic"), html.Th("Value")])),
            html.Tbody(table_rows)
        ], className='stats-table')
    
    except Exception as e:
        print(f"Error in stats table: {e}")
        return html.Div(f"Error generating statistics: {str(e)}")

@callback(
    Output('boxplot-comparison', 'figure'),
    Input('boxplot-variable', 'value')
)
def update_boxplot_comparison(variable):
    if not variable:
        return go.Figure().update_layout(
            title="No variable selected",
            template="plotly_white"
        )
    
    try:
        fig = go.Figure()
        
        # Add box plot for original data if it exists and has the variable
        if not original_data.empty and variable in original_data.columns:
            fig.add_trace(go.Box(
                y=original_data[variable].dropna(),
                name='Original',
                marker_color=colors['primary'],
                boxmean=True  # adds mean and standard deviation
            ))
        
        # Add box plots for synthetic data
        for i, method in enumerate(methods):
            if method in synthetic_data and variable in synthetic_data[method].columns:
                fig.add_trace(go.Box(
                    y=synthetic_data[method][variable].dropna(),
                    name=method,
                    marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)],
                    boxmean=True
                ))
        
        fig.update_layout(
            title=f'Box Plot Comparison for {variable}',
            yaxis_title=variable,
            template="plotly_white",
            boxmode='group'
        )
        
        return fig
    except Exception as e:
        print(f"Error in box plot: {e}")
        return go.Figure().update_layout(
            title=f"Error generating box plot: {str(e)}",
            template="plotly_white"
        )

@callback(
    Output('pair-plot', 'figure'),
    Input('pairplot-method', 'value'),
    Input('pairplot-variables', 'value')
)
def update_pair_plot(method, variables):
    if not variables or len(variables) < 2:
        return go.Figure().update_layout(
            title="Please select at least 2 variables for pair plot",
            template="plotly_white"
        )
    
    if len(variables) > 4:
        variables = variables[:4]  # Limit to 4 variables for performance
    
    try:
        if method == 'Original':
            data = original_data
        else:
            data = synthetic_data.get(method, pd.DataFrame())
        
        if data.empty:
            return go.Figure().update_layout(
                title="No data available",
                template="plotly_white"
            )
        
        # Create pair plot
        fig = px.scatter_matrix(
            data[variables].dropna(),
            dimensions=variables,
            title=f'Pair Plot for {method} Data',
            opacity=0.7,
            template="plotly_white"
        )
        
        # Update layout for better readability
        fig.update_layout(
            height=600,
            width=800
        )
        
        # Update traces
        fig.update_traces(
            diagonal_visible=False,
            showupperhalf=False,
            marker=dict(size=5, color=colors['primary'])
        )
        
        return fig
    except Exception as e:
        print(f"Error in pair plot: {e}")
        return go.Figure().update_layout(
            title=f"Error generating pair plot: {str(e)}",
            template="plotly_white"
        )

@callback(
    Output('violin-comparison', 'figure'),
    Input('violin-variable', 'value')
)
def update_violin_comparison(variable):
    if not variable:
        return go.Figure().update_layout(
            title="No variable selected",
            template="plotly_white"
        )
    
    try:
        # Prepare data for violin plot
        violin_data = []
        
        # Add original data if it exists and has the variable
        if not original_data.empty and variable in original_data.columns:
            orig_data = original_data[variable].dropna()
            violin_data.append({
                'y': orig_data,
                'type': 'violin',
                'name': 'Original',
                'box': {'visible': True},
                'meanline': {'visible': True},
                'line': {'color': colors['primary']},
                'fillcolor': px.colors.qualitative.Plotly[0],
                'opacity': 0.6
            })
        
        # Add synthetic data
        for i, method in enumerate(methods):
            if method in synthetic_data and variable in synthetic_data[method].columns:
                syn_data = synthetic_data[method][variable].dropna()
                violin_data.append({
                    'y': syn_data,
                    'type': 'violin',
                    'name': method,
                    'box': {'visible': True},
                    'meanline': {'visible': True},
                    'line': {'color': px.colors.qualitative.Plotly[(i+1) % len(px.colors.qualitative.Plotly)]},
                    'fillcolor': px.colors.qualitative.Plotly[(i+1) % len(px.colors.qualitative.Plotly)],
                    'opacity': 0.6
                })
        
        fig = go.Figure(data=violin_data)
        
        fig.update_layout(
            title=f'Violin Plot Comparison for {variable}',
            yaxis_title=variable,
            violinmode='group',
            template="plotly_white"
        )
        
        return fig
    except Exception as e:
        print(f"Error in violin plot: {e}")
        return go.Figure().update_layout(
            title=f"Error generating violin plot: {str(e)}",
            template="plotly_white"
        )

@callback(
    Output('cdf-plot', 'figure'),
    Input('cdf-variable', 'value')
)
def update_cdf_plot(variable):
    if not variable:
        return go.Figure().update_layout(
            title="No variable selected",
            template="plotly_white"
        )
    
    try:
        fig = go.Figure()
        
        # Add CDF for original data if it exists and has the variable
        if not original_data.empty and variable in original_data.columns:
            orig_data = original_data[variable].dropna().sort_values()
            y_orig = np.arange(1, len(orig_data) + 1) / len(orig_data)
            fig.add_trace(go.Scatter(
                x=orig_data,
                y=y_orig,
                mode='lines',
                name='Original',
                line=dict(color=colors['primary'], width=2)
            ))
        
        # Add CDFs for synthetic data
        for i, method in enumerate(methods):
            if method in synthetic_data and variable in synthetic_data[method].columns:
                syn_data = synthetic_data[method][variable].dropna().sort_values()
                y_syn = np.arange(1, len(syn_data) + 1) / len(syn_data)
                fig.add_trace(go.Scatter(
                    x=syn_data,
                    y=y_syn,
                    mode='lines',
                    name=method,
                    line=dict(color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)], width=2)
                ))
        
        fig.update_layout(
            title=f'Cumulative Distribution Function for {variable}',
            xaxis_title=variable,
            yaxis_title='Cumulative Probability',
            template="plotly_white"
        )
        
        return fig
    except Exception as e:
        print(f"Error in CDF plot: {e}")
        return go.Figure().update_layout(
            title=f"Error generating CDF plot: {str(e)}",
            template="plotly_white"
        )

@callback(
    Output('categorical-comparison', 'figure'),
    Input('categorical-variable', 'value')
)
def update_categorical_comparison(variable):
    if not variable:
        return go.Figure().update_layout(
            title="No variable selected",
            template="plotly_white"
        )
    
    try:
        fig = go.Figure()
        
        # Add bar chart for original data if it exists and has the variable
        if not original_data.empty and variable in original_data.columns:
            orig_counts = original_data[variable].value_counts(normalize=True).reset_index()
            orig_counts.columns = [variable, 'count']
            fig.add_trace(go.Bar(
                x=orig_counts[variable],
                y=orig_counts['count'],
                name='Original',
                marker_color=colors['primary']
            ))
        
        # Add bar charts for synthetic data
        for i, method in enumerate(methods):
            if method in synthetic_data and variable in synthetic_data[method].columns:
                syn_counts = synthetic_data[method][variable].value_counts(normalize=True).reset_index()
                syn_counts.columns = [variable, 'count']
                fig.add_trace(go.Bar(
                    x=syn_counts[variable],
                    y=syn_counts['count'],
                    name=method,
                    marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                ))
        
        fig.update_layout(
            title=f'Categorical Distribution for {variable}',
            xaxis_title=variable,
            yaxis_title='Proportion',
            barmode='group',
            template="plotly_white"
        )
        
        return fig
    except Exception as e:
        print(f"Error in categorical comparison: {e}")
        return go.Figure().update_layout(
            title=f"Error generating categorical comparison: {str(e)}",
            template="plotly_white"
        )

@callback(
    Output('variable-importance', 'figure'),
    Input('importance-method', 'value')
)
def update_variable_importance(method):
    if method == 'Original':
        data = original_data
    else:
        data = synthetic_data.get(method, pd.DataFrame())
    
    if data.empty:
        return go.Figure().update_layout(
            title="No data available",
            template="plotly_white"
        )
    
    try:
        # Calculate variable importance based on variance
        num_cols = [col for col in numerical_cols if col in data.columns]
        
        if not num_cols:
            return go.Figure().update_layout(
                title="No numerical columns available",
                template="plotly_white"
            )
        
        # Calculate variance for each column
        variances = data[num_cols].var().sort_values(ascending=False)
        
        # Create bar chart
        fig = go.Figure(go.Bar(
            x=variances.index,
            y=variances.values,
            marker_color=colors['primary']
        ))
        
        fig.update_layout(
            title=f'Variable Importance (Variance) for {method} Data',
            xaxis_title='Variable',
            yaxis_title='Variance',
            template="plotly_white",
            xaxis=dict(tickangle=-45)
        )
        
        return fig
    except Exception as e:
        print(f"Error in variable importance: {e}")
        return go.Figure().update_layout(
            title=f"Error generating variable importance: {str(e)}",
            template="plotly_white"
        )

@callback(
    Output('missing-values-plot', 'figure'),
    Input('missing-method', 'value')
)
def update_missing_values_plot(method):
    if method == 'Original':
        data = original_data
    else:
        data = synthetic_data.get(method, pd.DataFrame())
    
    if data.empty:
        return go.Figure().update_layout(
            title="No data available",
            template="plotly_white"
        )
    
    try:
        # Calculate missing values percentage
        missing_pct = data.isnull().mean() * 100
        missing_pct = missing_pct.sort_values(ascending=False)
        
        # Filter out columns with no missing values
        missing_pct = missing_pct[missing_pct > 0]
        
        if missing_pct.empty:
            return go.Figure().update_layout(
                title=f"No missing values in {method} data",
                template="plotly_white"
            )
        
        # Create bar chart
        fig = go.Figure(go.Bar(
            x=missing_pct.index,
            y=missing_pct.values,
            marker_color=colors['accent']
        ))
        
        fig.update_layout(
            title=f'Missing Values Analysis for {method} Data',
            xaxis_title='Column',
            yaxis_title='Missing Values (%)',
            template="plotly_white",
            xaxis=dict(tickangle=-45)
        )
        
        return fig
    except Exception as e:
        print(f"Error in missing values plot: {e}")
        return go.Figure().update_layout(
            title=f"Error generating missing values plot: {str(e)}",
            template="plotly_white"
        )

@callback(
    Output('quality-content', 'children'),
    Input('tsne-methods', 'value')
)
def update_quality_metrics(selected_methods):
    if not selected_methods or 'Original' not in selected_methods or len(selected_methods) < 2:
        return html.P("Please select Original and at least one synthetic data method in the t-SNE visualization to see quality metrics.")
    
    try:
        metrics_tables = []
        
        for method in selected_methods:
            if method == 'Original' or method not in synthetic_data:
                continue
            
            # Calculate basic quality metrics
            metrics = {}
            
            # Common columns
            common_cols = [col for col in numerical_cols if col in original_data.columns and col in synthetic_data[method].columns]
            
            if not common_cols:
                metrics_tables.append(html.Div([
                    html.H4(f"{method} vs Original", style={'color': colors['secondary']}),
                    html.P("No common numerical columns found for comparison")
                ]))
                continue
            
            # Calculate metrics for each column
            col_metrics = []
            for col in common_cols:
                orig_values = original_data[col].dropna()
                syn_values = synthetic_data[method][col].dropna()
                
                if len(orig_values) == 0 or len(syn_values) == 0:
                    continue
                
                # Mean difference
                mean_diff = abs(orig_values.mean() - syn_values.mean())
                mean_diff_pct = mean_diff / abs(orig_values.mean()) * 100 if orig_values.mean() != 0 else float('inf')
                
                # Standard deviation difference
                std_diff = abs(orig_values.std() - syn_values.std())
                std_diff_pct = std_diff / orig_values.std() * 100 if orig_values.std() != 0 else float('inf')
                
                # Min-max difference
                min_diff = abs(orig_values.min() - syn_values.min())
                max_diff = abs(orig_values.max() - syn_values.max())
                
                col_metrics.append({
                    'Column': col,
                    'Mean Diff %': f"{mean_diff_pct:.2f}%",
                    'Std Diff %': f"{std_diff_pct:.2f}%",
                    'Min Diff': f"{min_diff:.2f}",
                    'Max Diff': f"{max_diff:.2f}"
                })
            
            # Create table
            if col_metrics:
                table_header = [
                    html.Thead(html.Tr([
                        html.Th("Column"),
                        html.Th("Mean Diff %"),
                        html.Th("Std Diff %"),
                        html.Th("Min Diff"),
                        html.Th("Max Diff")
                    ]))
                ]
                
                table_rows = []
                for metric in col_metrics:
                    table_rows.append(html.Tr([
                        html.Td(metric['Column']),
                        html.Td(metric['Mean Diff %']),
                        html.Td(metric['Std Diff %']),
                        html.Td(metric['Min Diff']),
                        html.Td(metric['Max Diff'])
                    ]))
                
                metrics_tables.append(html.Div([
                    html.H4(f"{method} vs Original", style={'color': colors['secondary']}),
                    html.Table(table_header + [html.Tbody(table_rows)], style={'width': '100%'})
                ], style={'marginBottom': '20px'}))
            else:
                metrics_tables.append(html.Div([
                    html.H4(f"{method} vs Original", style={'color': colors['secondary']}),
                    html.P("Could not calculate metrics for any common columns")
                ]))
        
        if not metrics_tables:
            return html.P("No metrics could be calculated. Please ensure that Original and synthetic data have common numerical columns.")
        
        return metrics_tables
    
    except Exception as e:
        print(f"Error in quality metrics: {e}")
        return html.P(f"Error calculating quality metrics: {str(e)}")

@callback(
    Output('age-distribution', 'figure'),
    Input('tsne-methods', 'value')  # Using the same input as t-SNE to trigger this callback
)
def update_age_distribution(selected_methods):
    try:
        fig = go.Figure()
        
        # Add histogram for original data
        if not original_data.empty and 'age' in original_data.columns:
            fig.add_trace(go.Histogram(
                x=original_data['age'].dropna(),
                name='Original',
                opacity=0.7,
                nbinsx=30,
                marker_color=colors['primary']
            ))
        
        # Add histograms for synthetic data
        for i, method in enumerate(methods):
            if method in synthetic_data and 'age' in synthetic_data[method].columns:
                fig.add_trace(go.Histogram(
                    x=synthetic_data[method]['age'].dropna(),
                    name=method,
                    opacity=0.7,
                    nbinsx=30,
                    marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                ))
        
        fig.update_layout(
            barmode='overlay',
            title='Age Distribution Comparison',
            xaxis_title='Age',
            yaxis_title='Count',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white"
        )
        
        return fig
    except Exception as e:
        print(f"Error in age distribution plot: {e}")
        return go.Figure().update_layout(
            title=f"Error generating age distribution plot: {str(e)}",
            template="plotly_white"
        )

@callback(
    Output('scatter-comparison', 'figure'),
    Input('scatter-x-variable', 'value'),
    Input('scatter-y-variable', 'value')
)
def update_scatter_comparison(x_variable, y_variable):
    if not x_variable or not y_variable:
        return go.Figure().update_layout(
            title="Please select both X and Y variables",
            template="plotly_white"
        )
    
    try:
        fig = go.Figure()
        
        # Add scatter for original data
        if not original_data.empty and x_variable in original_data.columns and y_variable in original_data.columns:
            fig.add_trace(go.Scatter(
                x=original_data[x_variable].dropna(),
                y=original_data[y_variable].dropna(),
                mode='markers',
                name='Original',
                marker=dict(
                    color=colors['primary'],
                    size=8,
                    opacity=0.6
                )
            ))
        
        # Add scatter for synthetic data
        for i, method in enumerate(methods):
            if method in synthetic_data and x_variable in synthetic_data[method].columns and y_variable in synthetic_data[method].columns:
                fig.add_trace(go.Scatter(
                    x=synthetic_data[method][x_variable].dropna(),
                    y=synthetic_data[method][y_variable].dropna(),
                    mode='markers',
                    name=method,
                    marker=dict(
                        color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)],
                        size=8,
                        opacity=0.6
                    )
                ))
        
        fig.update_layout(
            title=f'Scatter Plot: {y_variable} vs {x_variable}',
            xaxis_title=x_variable,
            yaxis_title=y_variable,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        print(f"Error in scatter plot: {e}")
        return go.Figure().update_layout(
            title=f"Error generating scatter plot: {str(e)}",
            template="plotly_white"
        )

@callback(
    Output('completeness-comparison', 'figure'),
    Input('tsne-methods', 'value')  # Adding an input to trigger the callback
)
def update_completeness_comparison(selected_methods):
    try:
        # Calculate completeness for each dataset
        completeness_data = []
        
        # Original data
        if not original_data.empty:
            orig_completeness = (1 - original_data.isnull().mean()) * 100
            completeness_data.append({
                'Method': 'Original',
                'Columns': orig_completeness.index.tolist(),
                'Completeness': orig_completeness.values.tolist()
            })
        
        # Synthetic data
        for method in methods:
            if method in synthetic_data and not synthetic_data[method].empty:
                syn_completeness = (1 - synthetic_data[method].isnull().mean()) * 100
                completeness_data.append({
                    'Method': method,
                    'Columns': syn_completeness.index.tolist(),
                    'Completeness': syn_completeness.values.tolist()
                })
        
        if not completeness_data:
            return go.Figure().update_layout(
                title="No data available for completeness comparison",
                template="plotly_white"
            )
        
        # Create heatmap
        fig = go.Figure()
        
        for i, data in enumerate(completeness_data):
            for j, (col, val) in enumerate(zip(data['Columns'], data['Completeness'])):
                fig.add_trace(go.Bar(
                    x=[col],
                    y=[val],
                    name=data['Method'],
                    marker_color=colors['primary'] if data['Method'] == 'Original' else 
                              px.colors.qualitative.Plotly[(i-1) % len(px.colors.qualitative.Plotly)],
                    showlegend=j == 0  # Show legend only once per method
                ))
        
        fig.update_layout(
            title='Data Completeness Comparison (% of non-null values)',
            xaxis_title='Column',
            yaxis_title='Completeness (%)',
            barmode='group',
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        print(f"Error in completeness comparison: {e}")
        return go.Figure().update_layout(
            title=f"Error generating completeness comparison: {str(e)}",
            template="plotly_white"
        )

# Run the app
if __name__ == '__main__':
    app.run(debug=True)