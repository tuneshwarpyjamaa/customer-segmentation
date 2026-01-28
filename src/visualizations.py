"""
Visualization Module

Generate insightful charts and plots for RFM analysis.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class RFMVisualizer:
    """
    Visualization utilities for RFM analysis.
    """
    
    def __init__(self, output_dir: str = "outputs/figures"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color palette
        self.colors = px.colors.qualitative.Set3
        
    def plot_segment_distribution(self, rfm: pd.DataFrame, save: bool = True) -> go.Figure:
        """
        Create pie chart showing customer distribution across segments.
        
        Args:
            rfm: DataFrame with RFM segments
            save: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        segment_counts = rfm['Segment'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=segment_counts.index,
            values=segment_counts.values,
            hole=0.4,
            marker=dict(colors=self.colors),
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title={
                'text': 'Customer Segment Distribution',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            showlegend=True,
            height=600,
            font=dict(size=12)
        )
        
        if save:
            fig.write_html(self.output_dir / "segment_distribution.html")
            print(f"  ✓ Saved segment_distribution.html")
        
        return fig
    
    def plot_segment_revenue(self, rfm: pd.DataFrame, save: bool = True) -> go.Figure:
        """
        Create bar chart showing revenue contribution by segment.
        
        Args:
            rfm: DataFrame with RFM segments
            save: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        segment_revenue = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
        
        fig = go.Figure(data=[go.Bar(
            x=segment_revenue.index,
            y=segment_revenue.values,
            marker=dict(
                color=segment_revenue.values,
                colorscale='Viridis',
                showscale=True
            ),
            text=[f'${val:,.0f}' for val in segment_revenue.values],
            textposition='outside'
        )])
        
        fig.update_layout(
            title={
                'text': 'Revenue Contribution by Customer Segment',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            xaxis_title='Customer Segment',
            yaxis_title='Total Revenue ($)',
            height=600,
            showlegend=False
        )
        
        if save:
            fig.write_html(self.output_dir / "segment_revenue.html")
            print(f"  ✓ Saved segment_revenue.html")
        
        return fig
    
    def plot_rfm_3d_scatter(self, rfm: pd.DataFrame, save: bool = True) -> go.Figure:
        """
        Create 3D scatter plot of RFM scores.
        
        Args:
            rfm: DataFrame with RFM scores
            save: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=[go.Scatter3d(
            x=rfm['R_Score'],
            y=rfm['F_Score'],
            z=rfm['M_Score'],
            mode='markers',
            marker=dict(
                size=5,
                color=rfm['RFM_Value'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="RFM Value"),
                opacity=0.7
            ),
            text=rfm['Segment'],
            hovertemplate='<b>%{text}</b><br>' +
                          'Recency Score: %{x}<br>' +
                          'Frequency Score: %{y}<br>' +
                          'Monetary Score: %{z}<br>' +
                          '<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': '3D RFM Score Distribution',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            scene=dict(
                xaxis_title='Recency Score',
                yaxis_title='Frequency Score',
                zaxis_title='Monetary Score'
            ),
            height=700
        )
        
        if save:
            fig.write_html(self.output_dir / "rfm_3d_scatter.html")
            print(f"  ✓ Saved rfm_3d_scatter.html")
        
        return fig
    
    def plot_rfm_heatmap(self, rfm: pd.DataFrame, save: bool = True) -> go.Figure:
        """
        Create heatmap showing correlation between RFM metrics.
        
        Args:
            rfm: DataFrame with RFM metrics
            save: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        # Calculate correlation matrix
        corr_data = rfm[['Recency', 'Frequency', 'Monetary']].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_data.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 16},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title={
                'text': 'RFM Metrics Correlation Heatmap',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            height=500,
            width=600
        )
        
        if save:
            fig.write_html(self.output_dir / "rfm_heatmap.html")
            print(f"  ✓ Saved rfm_heatmap.html")
        
        return fig
    
    def plot_segment_metrics(self, rfm: pd.DataFrame, save: bool = True) -> go.Figure:
        """
        Create subplots showing average metrics per segment.
        
        Args:
            rfm: DataFrame with RFM segments
            save: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        # Calculate averages by segment
        segment_avg = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Average Recency (days)', 'Average Frequency', 'Average Monetary ($)')
        )
        
        # Recency
        fig.add_trace(
            go.Bar(x=segment_avg.index, y=segment_avg['Recency'], 
                   marker_color='lightblue', name='Recency'),
            row=1, col=1
        )
        
        # Frequency
        fig.add_trace(
            go.Bar(x=segment_avg.index, y=segment_avg['Frequency'],
                   marker_color='lightgreen', name='Frequency'),
            row=1, col=2
        )
        
        # Monetary
        fig.add_trace(
            go.Bar(x=segment_avg.index, y=segment_avg['Monetary'],
                   marker_color='lightcoral', name='Monetary'),
            row=1, col=3
        )
        
        fig.update_layout(
            title={
                'text': 'Average RFM Metrics by Segment',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            height=500,
            showlegend=False
        )
        
        # Rotate x-axis labels
        fig.update_xaxes(tickangle=45)
        
        if save:
            fig.write_html(self.output_dir / "segment_metrics.html")
            print(f"  ✓ Saved segment_metrics.html")
        
        return fig
    
    def plot_customer_distribution(self, df: pd.DataFrame, save: bool = True) -> go.Figure:
        """
        Plot customer and revenue distribution over time.
        
        Args:
            df: Transaction DataFrame
            save: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        # Aggregate by month
        df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
        monthly_data = df.groupby('YearMonth').agg({
            'CustomerID': 'nunique',
            'TotalAmount': 'sum'
        }).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add customers trace
        fig.add_trace(
            go.Scatter(x=monthly_data['YearMonth'], y=monthly_data['CustomerID'],
                      name='Unique Customers', line=dict(color='blue', width=3)),
            secondary_y=False
        )
        
        # Add revenue trace
        fig.add_trace(
            go.Scatter(x=monthly_data['YearMonth'], y=monthly_data['TotalAmount'],
                      name='Total Revenue', line=dict(color='green', width=3)),
            secondary_y=True
        )
        
        fig.update_layout(
            title={
                'text': 'Customer and Revenue Trends Over Time',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            height=500,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text='Month', tickangle=45)
        fig.update_yaxes(title_text='Unique Customers', secondary_y=False)
        fig.update_yaxes(title_text='Total Revenue ($)', secondary_y=True)
        
        if save:
            fig.write_html(self.output_dir / "time_trends.html")
            print(f"  ✓ Saved time_trends.html")
        
        return fig
    
    def plot_top_products(self, df: pd.DataFrame, top_n: int = 10, save: bool = True) -> go.Figure:
        """
        Plot top products by revenue.
        
        Args:
            df: Transaction DataFrame
            top_n: Number of top products to show
            save: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        # Get top products
        product_revenue = df.groupby('Description')['TotalAmount'].sum().nlargest(top_n)
        
        fig = go.Figure(data=[go.Bar(
            x=product_revenue.values,
            y=product_revenue.index,
            orientation='h',
            marker=dict(
                color=product_revenue.values,
                colorscale='Blues',
                showscale=False
            ),
            text=[f'${val:,.0f}' for val in product_revenue.values],
            textposition='outside'
        )])
        
        fig.update_layout(
            title={
                'text': f'Top {top_n} Products by Revenue',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            xaxis_title='Total Revenue ($)',
            yaxis_title='Product',
            height=600
        )
        
        if save:
            fig.write_html(self.output_dir / "top_products.html")
            print(f"  ✓ Saved top_products.html")
        
        return fig


def create_all_visualizations(df: pd.DataFrame, rfm: pd.DataFrame, 
                              output_dir: str = "outputs/figures") -> None:
    """
    Generate all visualizations at once.
    
    Args:
        df: Transaction DataFrame
        rfm: RFM DataFrame with segments
        output_dir: Output directory
    """
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    visualizer = RFMVisualizer(output_dir)
    
    # Create all plots
    visualizer.plot_segment_distribution(rfm)
    visualizer.plot_segment_revenue(rfm)
    visualizer.plot_rfm_3d_scatter(rfm)
    visualizer.plot_rfm_heatmap(rfm)
    visualizer.plot_segment_metrics(rfm)
    visualizer.plot_customer_distribution(df)
    visualizer.plot_top_products(df)
    
    print("\n" + "="*80)
    print(f"All visualizations saved to {output_dir}/")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Test visualizations
    from pathlib import Path
    from data_loader import load_data_optimized
    from data_cleaner import clean_data
    from rfm_analysis import calculate_rfm, segment_customers
    
    data_path = Path(__file__).parent.parent / "data" / "raw" / "OnlineRetail.xlsx"
    
    if data_path.exists():
        df = load_data_optimized(str(data_path))
        cleaned_df = clean_data(df)
        rfm = calculate_rfm(cleaned_df)
        rfm_segmented = segment_customers(rfm)
        
        create_all_visualizations(cleaned_df, rfm_segmented)
    else:
        print(f"Data file not found at {data_path}")
