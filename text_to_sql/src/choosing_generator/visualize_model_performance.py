import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
import argparse
import glob
import os

def load_evaluation_data(csv_pattern="batch_evaluation_*.csv"):
    """Load and combine evaluation data from CSV files"""
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found matching pattern: {csv_pattern}")
        return None
    
    print(f"Found {len(csv_files)} evaluation files:")
    for file in csv_files:
        print(f"  - {file}")
    
    # Combine all CSV files
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"\nLoaded {len(combined_df)} total evaluations")
    
    return combined_df

def aggregate_by_model(df):
    """Aggregate performance metrics by model"""
    agg_metrics = df.groupby('model_name').agg({
        'results_match': ['mean', 'std', 'count'],
        'latency_seconds': ['mean', 'median', 'std'],
        'actual_cost': ['mean', 'median', 'std'],
        'query_successful': 'mean',
        'total_tokens': ['mean', 'median'],
        'db_name': 'first'  # Keep track of which db was used
    }).round(6)
    
    # Flatten column names
    agg_metrics.columns = ['_'.join(col).strip() for col in agg_metrics.columns]
    agg_metrics = agg_metrics.reset_index()
    
    return agg_metrics

def standardize_metrics(df, method='minmax'):
    """Standardize metrics for fair comparison"""
    metrics_to_standardize = [
        'results_match_mean',
        'latency_seconds_median', 
        'actual_cost_median'
    ]
    
    df_copy = df.copy()
    
    if method == 'minmax':
        scaler = MinMaxScaler()
        # For latency and cost, we want lower to be better, so we'll invert them
        df_copy['accuracy_score'] = df_copy['results_match_mean']
        df_copy['speed_score'] = 1 - scaler.fit_transform(df_copy[['latency_seconds_median']])[:, 0]
        df_copy['cost_efficiency_score'] = 1 - scaler.fit_transform(df_copy[['actual_cost_median']])[:, 0]
        
    elif method == 'zscore':
        scaler = StandardScaler()
        standardized = scaler.fit_transform(df_copy[metrics_to_standardize])
        df_copy['accuracy_score'] = standardized[:, 0]
        df_copy['speed_score'] = -standardized[:, 1]  # Invert latency (lower is better)
        df_copy['cost_efficiency_score'] = -standardized[:, 2]  # Invert cost (lower is better)
    
    return df_copy

def create_3d_scatter_matplotlib(df, title="Model Performance Trade-offs"):
    """Create 3D scatter plot using matplotlib"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot
    scatter = ax.scatter(
        df['accuracy_score'], 
        df['speed_score'], 
        df['cost_efficiency_score'],
        s=df['results_match_mean'] * 1000,  # Size based on accuracy
        c=df['latency_seconds_median'], 
        cmap='viridis',
        alpha=0.8,
        edgecolors='black',
        linewidth=1
    )
    
    # Add labels for each point
    for i, model in enumerate(df['model_name']):
        ax.text(
            df['accuracy_score'].iloc[i],
            df['speed_score'].iloc[i], 
            df['cost_efficiency_score'].iloc[i],
            f'  {model.split("/")[-1]}',  # Show just model name, not provider
            fontsize=9
        )
    
    # Set labels and title
    ax.set_xlabel('Accuracy Score (Standardized)', fontsize=12)
    ax.set_ylabel('Speed Score (Standardized)', fontsize=12)
    ax.set_zlabel('Cost Efficiency Score (Standardized)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    plt.colorbar(scatter, label='Median Latency (seconds)', shrink=0.8)
    
    # Add legend for size
    sizes = [0.2, 0.3, 0.4, 0.5]
    size_labels = ['20%', '30%', '40%', '50%']
    legend_elements = [plt.scatter([], [], s=s*1000, c='gray', alpha=0.8, edgecolors='black') 
                      for s in sizes]
    ax.legend(legend_elements, size_labels, title='Accuracy Rate', 
             loc='upper left', bbox_to_anchor=(0, 1))
    
    plt.tight_layout()
    return fig

def create_3d_scatter_plotly(df, title="Interactive Model Performance Analysis"):
    """Create interactive 3D scatter plot using plotly"""
    
    fig = go.Figure(data=[go.Scatter3d(
        x=df['accuracy_score'],
        y=df['speed_score'],
        z=df['cost_efficiency_score'],
        mode='markers+text',
        marker=dict(
            size=df['results_match_mean'] * 30 + 10,  # Size based on accuracy
            color=df['latency_seconds_median'],
            colorscale='Viridis',
            colorbar=dict(title="Median Latency (s)"),
            opacity=0.8,
            line=dict(width=2, color='black')
        ),
        text=[model.split('/')[-1] for model in df['model_name']],  # Just model names
        textposition="top center",
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "Accuracy: %{customdata[0]:.1%}<br>" +
            "Median Latency: %{customdata[1]:.2f}s<br>" +
            "Median Cost: $%{customdata[2]:.6f}<br>" +
            "Success Rate: %{customdata[3]:.1%}<br>" +
            "Median Tokens: %{customdata[4]:.0f}<br>" +
            "<extra></extra>"
        ),
        customdata=np.column_stack([
            df['results_match_mean'],
            df['latency_seconds_median'], 
            df['actual_cost_median'],
            df['query_successful_mean'],
            df['total_tokens_median']
        ])
    )])
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        scene=dict(
            xaxis_title='Accuracy Score (Standardized)',
            yaxis_title='Speed Score (Standardized)', 
            zaxis_title='Cost Efficiency Score (Standardized)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=900,
        height=700,
        margin=dict(r=20, b=10, l=10, t=40)
    )
    
    return fig

def create_performance_dashboard(df):
    """Create a comprehensive performance dashboard"""
    fig, axes = plt.subplots(2, 2, figsize=(28, 16), gridspec_kw={'width_ratios': [1.5, 1.5], 'height_ratios': [1, 1]})
    
    # Increase base font sizes
    plt.rcParams.update({'font.size': 14})
    
    # 1. Accuracy vs Latency
    axes[0, 0].scatter(df['latency_seconds_median'], df['results_match_mean'], 
                      s=df['actual_cost_median']*1000000, alpha=0.7, edgecolors='black', linewidth=2)
    axes[0, 0].set_xlabel('Median Latency (seconds)', fontsize=16, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy Rate', fontsize=16, fontweight='bold')
    axes[0, 0].set_title('Accuracy vs Latency\n(Bubble size = Cost)', fontsize=18, fontweight='bold')
    axes[0, 0].tick_params(axis='both', which='major', labelsize=14)
    for i, model in enumerate(df['model_name']):
        axes[0, 0].annotate(model.split('/')[-1], 
                           (df['latency_seconds_median'].iloc[i], df['results_match_mean'].iloc[i]),
                           xytext=(8, 8), textcoords='offset points', fontsize=16, fontweight='bold')
    
    # 2. Cost vs Accuracy
    axes[0, 1].scatter(df['actual_cost_median'], df['results_match_mean'],
                      s=df['latency_seconds_median']*50, alpha=0.7, edgecolors='black', linewidth=2)
    axes[0, 1].set_xlabel('Median Cost ($)', fontsize=16, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy Rate', fontsize=16, fontweight='bold')
    axes[0, 1].set_title('Cost vs Accuracy\n(Bubble size = Latency)', fontsize=18, fontweight='bold')
    axes[0, 1].tick_params(axis='both', which='major', labelsize=14)
    for i, model in enumerate(df['model_name']):
        axes[0, 1].annotate(model.split('/')[-1],
                           (df['actual_cost_median'].iloc[i], df['results_match_mean'].iloc[i]),
                           xytext=(8, 8), textcoords='offset points', fontsize=16, fontweight='bold')
    
    # 3. Overall Performance Radar - Use raw values with fixed meaningful ranges
    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    axes[1, 0].remove()
    ax_radar = fig.add_subplot(2, 2, 3, projection='polar')

    SLOW = 10  # 20 seconds
    COSTLY = 0.001 # $0.001
    
    for i, model in enumerate(df['model_name']):
        # Use meaningful fixed ranges for each metric
        # Accuracy: 0-1 (natural percentage)
        accuracy_val = df['results_match_mean'].iloc[i]
        
        # Speed: Convert latency to speed score (0-1, where 1 = instant, 0 = 10+ seconds)
        latency = df['latency_seconds_median'].iloc[i]
        speed_val = max(0, min(1, (SLOW - latency) / SLOW))  # 0 seconds = 1.0, 10+ seconds = 0.0
        
        # Cost efficiency: Convert cost to efficiency (0-1, where 1 = free, 0 = $0.001+)
        cost = df['actual_cost_median'].iloc[i]
        cost_efficiency_val = max(0, min(1, (COSTLY - cost) / COSTLY))  # $0 = 1.0, $0.001+ = 0.0
        
        values = [accuracy_val, speed_val, cost_efficiency_val]
        values += values[:1]  # Complete the circle
        
        ax_radar.plot(angles, values, 'o-', linewidth=4, 
                     label=f"{model.split('/')[-1]}", alpha=0.8, markersize=12)
        ax_radar.fill(angles, values, alpha=0.15)
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(['Accuracy\n(0=0%, 1=100%)', f'Speed\n(0={SLOW}s+, 1=instant)', f'Cost Efficiency\n(0=${COSTLY}+, 1=free)'], fontsize=14, fontweight='bold')
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=14)
    ax_radar.tick_params(axis='both', which='major', labelsize=14)
    ax_radar.grid(True, linewidth=1.5)
    ax_radar.set_title('Performance Profile', y=1.08, fontsize=18, fontweight='bold')
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=14, frameon=True, fancybox=True, shadow=True)
    
    # 4. Summary Statistics Table
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    summary_data = []
    for _, row in df.iterrows():
        summary_data.append([
            row['model_name'].split('/')[-1],
            f"{row['results_match_mean']:.1%}",
            f"{row['latency_seconds_median']:.2f}s",
            f"${row['actual_cost_median']:.6f}",
            f"{row['query_successful_mean']:.1%}"
        ])
    table = axes[1, 1].table(cellText=summary_data,
                            colLabels=['Model', 'Accuracy', 'Latency', 'Cost', 'Success Rate'],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(17)
    table.scale(3.5, 3.5)
    
    # Style the table header
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_width(0.35)
        table[(0, i)].set_height(0.15)
    
    # Style the table cells
    for i in range(1, len(summary_data) + 1):
        for j in range(len(summary_data[0])):
            table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
            table[(i, j)].set_text_props(weight='bold')
            table[(i, j)].set_width(0.35)
            table[(i, j)].set_height(0.15)
    
    axes[1, 1].set_title('Performance Summary', fontweight='bold', fontsize=18, pad=20)
    
    plt.tight_layout(pad=3.0)
    return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize model performance in 3D")
    parser.add_argument("--csv-pattern", default="batch_evaluation_*.csv",
                       help="Pattern to match CSV files")
    parser.add_argument("--standardization", choices=['minmax', 'zscore'], 
                       default='minmax', help="Standardization method")
    parser.add_argument("--output-dir", default=".", 
                       help="Directory to save plots")
    parser.add_argument("--interactive", action="store_true",
                       help="Create interactive plotly visualization")
    
    args = parser.parse_args()
    
    # Load and process data
    df = load_evaluation_data(args.csv_pattern)
    if df is None:
        return
    
    # Aggregate by model
    agg_df = aggregate_by_model(df)
    agg_df['accuracy_score'] = agg_df['results_match_mean']
    agg_df['speed_score'] = 1 - (agg_df['latency_seconds_median'] / 10)
    agg_df['cost_efficiency_score'] = 1 - (agg_df['actual_cost_median'] / 0.001)
    print(f"\nModel performance summary:")
    print(agg_df[['model_name', 'results_match_mean', 'latency_seconds_median', 'actual_cost_median']])
    
    # Standardize metrics
    standardized_df = standardize_metrics(agg_df, method=args.standardization)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    
    # 3D Matplotlib plot
    fig_3d = create_3d_scatter_matplotlib(agg_df)
    output_path = os.path.join(args.output_dir, "model_performance_3d.png")
    fig_3d.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D plot: {output_path}")
    
    # Performance dashboard
    fig_dashboard = create_performance_dashboard(standardized_df)
    dashboard_path = os.path.join(args.output_dir, "performance_dashboard.png")
    fig_dashboard.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    print(f"Saved dashboard: {dashboard_path}")
    
    # Interactive plotly (if requested)
    if args.interactive:
        fig_interactive = create_3d_scatter_plotly(standardized_df)
        interactive_path = os.path.join(args.output_dir, "interactive_performance.html")
        fig_interactive.write_html(interactive_path)
        print(f"Saved interactive plot: {interactive_path}")
    
    # Show plots
    plt.show()
    
    # Print analysis
    print(f"\n🔍 Performance Analysis:")
    print(f"📊 Standardization method: {args.standardization}")
    
    best_accuracy = standardized_df.loc[standardized_df['results_match_mean'].idxmax()]
    best_speed = standardized_df.loc[standardized_df['speed_score'].idxmax()]
    best_cost = standardized_df.loc[standardized_df['cost_efficiency_score'].idxmax()]
    
    print(f"🎯 Best Accuracy: {best_accuracy['model_name']} ({best_accuracy['results_match_mean']:.1%})")
    print(f"⚡ Fastest: {best_speed['model_name']} ({standardized_df.loc[standardized_df['speed_score'].idxmax(), 'latency_seconds_median']:.2f}s)")
    print(f"💰 Most Cost-Efficient: {best_cost['model_name']} (${standardized_df.loc[standardized_df['cost_efficiency_score'].idxmax(), 'actual_cost_median']:.6f})")

if __name__ == "__main__":
    main() 