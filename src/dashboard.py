"""
Interactive Dashboard for Customer Segmentation

Dash-based web application for exploring RFM analysis results.
"""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')

# Import our modules
from data_loader import load_data_optimized
from data_cleaner import clean_data
from rfm_analysis import calculate_rfm, segment_customers, get_segment_summary
from cohort_analysis import create_cohort_analysis


# Load data
print("Loading data for dashboard...")
DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "OnlineRetail.csv"

if not DATA_PATH.exists():
    print(f"Error: Data file not found at {DATA_PATH}")
    print("Please download the dataset first.")
    exit(1)

df = load_data_optimized(str(DATA_PATH), file_format='csv')
cleaned_df = clean_data(df, remove_outliers=False)
rfm = calculate_rfm(cleaned_df)
rfm_segmented = segment_customers(rfm)
segment_summary = get_segment_summary(rfm_segmented)

# Calculate cohort analysis
print("Calculating cohort analysis...")
cohort_results = create_cohort_analysis(cleaned_df, period='M')
cohort_retention = cohort_results['retention_matrix']
cohort_revenue = cohort_results['revenue_matrix']
cohort_summary = cohort_results['cohort_summary']

print("Data loaded successfully!\n")


# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "Customer Segmentation Dashboard"


# Define color scheme
COLORS = {
    'primary': '#2c3e50',
    'secondary': '#3498db',
    'success': '#27ae60',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'light': '#ecf0f1',
    'dark': '#34495e'
}


# Visualization functions
def create_segment_pie():
    """Create segment distribution pie chart."""
    segment_counts = rfm_segmented['Segment'].value_counts()

    fig = go.Figure(data=[go.Pie(
        labels=segment_counts.index,
        values=segment_counts.values,
        hole=0.4,
        marker=dict(colors=px.colors.qualitative.Set3)
    )])

    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_revenue_bar():
    """Create revenue by segment bar chart."""
    segment_revenue = rfm_segmented.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)

    fig = go.Figure(data=[go.Bar(
        x=segment_revenue.index,
        y=segment_revenue.values,
        marker=dict(color=segment_revenue.values, colorscale='Viridis'),
        text=[f'₹{val:,.0f}' for val in segment_revenue.values],
        textposition='outside'
    )])

    fig.update_layout(
        height=400,
        xaxis_title='Segment',
        yaxis_title='Total Revenue (₹)',
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    return fig


def create_3d_scatter():
    """Create 3D scatter plot of RFM scores."""
    fig = go.Figure(data=[go.Scatter3d(
        x=rfm_segmented['R_Score'],
        y=rfm_segmented['F_Score'],
        z=rfm_segmented['M_Score'],
        mode='markers',
        marker=dict(
            size=4,
            color=rfm_segmented['RFM_Value'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="RFM Value"),
            opacity=0.7
        ),
        text=rfm_segmented['Segment'],
        hovertemplate='<b>%{text}</b><br>R: %{x}<br>F: %{y}<br>M: %{z}<extra></extra>'
    )])

    fig.update_layout(
        height=600,
        scene=dict(
            xaxis_title='Recency Score',
            yaxis_title='Frequency Score',
            zaxis_title='Monetary Score'
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


def create_rfm_bubble():
    """Recency vs Frequency bubble sized by Monetary."""
    df = rfm_segmented.copy()
    # prevent extremely large bubble sizes by scaling
    fig = px.scatter(
        df,
        x='Recency',
        y='Frequency',
        size='Monetary',
        color='Segment',
        hover_data=['CustomerID', 'Monetary'],
        size_max=40,
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={'Recency': 'Days since last purchase', 'Frequency': 'Purchase count'}
    )
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        title='Recency vs Frequency (bubble = Monetary)'
    )
    return fig


def create_segment_table():
    """Create segment summary table."""
    table = dbc.Table.from_dataframe(
        segment_summary,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="table-sm"
    )
    return table


def create_time_trends():
    """Create time trends chart."""
    cleaned_df['YearMonth'] = cleaned_df['InvoiceDate'].dt.to_period('M').astype(str)
    monthly_data = cleaned_df.groupby('YearMonth').agg({
        'CustomerID': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=monthly_data['YearMonth'], y=monthly_data['CustomerID'],
                  name='Customers', line=dict(color='blue', width=2)),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=monthly_data['YearMonth'], y=monthly_data['TotalAmount'],
                  name='Revenue', line=dict(color='green', width=2)),
        secondary_y=True
    )

    fig.update_layout(
        height=400,
        hovermode='x unified',
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.update_xaxes(title_text='Month', tickangle=45)
    fig.update_yaxes(title_text='Customers', secondary_y=False)
    fig.update_yaxes(title_text='Revenue (₹)', secondary_y=True)

    return fig


def create_top_customers():
    """Create top customers chart."""
    top_20 = rfm_segmented.nlargest(20, 'Monetary')

    fig = go.Figure(data=[go.Bar(
        x=top_20['CustomerID'].astype(str),
        y=top_20['Monetary'],
        marker=dict(color=top_20['Monetary'], colorscale='Blues'),
        text=top_20['Segment'],
        hovertemplate='Customer: %{x}<br>Revenue: ₹%{y:,.0f}<br>Segment: %{text}<extra></extra>'
    )])

    fig.update_layout(
        height=400,
        xaxis_title='Customer ID',
        yaxis_title='Total Spending (₹)',
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    fig.update_xaxes(tickangle=45)

    return fig


def create_heatmap():
    """Create RFM correlation heatmap."""
    corr_data = rfm_segmented[['Recency', 'Frequency', 'Monetary']].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns,
        y=corr_data.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_data.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 14}
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_cohort_retention_heatmap():
    """Create cohort retention heatmap."""
    if cohort_retention.empty:
        return go.Figure()
    
    # Convert to percentages and round
    retention_pct = (cohort_retention * 100).round(1)
    
    fig = go.Figure(data=go.Heatmap(
        z=retention_pct.values,
        x=[f'Month {i}' for i in retention_pct.columns],
        y=[str(idx) for idx in retention_pct.index],
        colorscale='YlOrRd',
        text=retention_pct.values,
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        hovertemplate='Cohort: %{y}<br>Month: %{x}<br>Retention: %{z:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        height=500,
        xaxis_title='Period (Months)',
        yaxis_title='Acquisition Cohort',
        title='Customer Retention Rate by Cohort (%)',
        margin=dict(l=80, r=20, t=60, b=40)
    )
    
    return fig


def create_cohort_revenue_heatmap():
    """Create cohort revenue retention heatmap."""
    if cohort_revenue.empty:
        return go.Figure()
    
    # Convert to percentages and round
    revenue_pct = (cohort_revenue * 100).round(1)
    
    fig = go.Figure(data=go.Heatmap(
        z=revenue_pct.values,
        x=[f'Month {i}' for i in revenue_pct.columns],
        y=[str(idx) for idx in revenue_pct.index],
        colorscale='Blues',
        text=revenue_pct.values,
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        hovertemplate='Cohort: %{y}<br>Month: %{x}<br>Revenue Retention: %{z:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        height=500,
        xaxis_title='Period (Months)',
        yaxis_title='Acquisition Cohort',
        title='Revenue Retention Rate by Cohort (%)',
        margin=dict(l=80, r=20, t=60, b=40)
    )
    
    return fig


def create_cohort_comparison_chart():
    """Create cohort comparison line chart."""
    if cohort_retention.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Plot retention curves for each cohort
    colors = px.colors.qualitative.Set3
    
    for i, (cohort, row) in enumerate(cohort_retention.iterrows()):
        fig.add_trace(go.Scatter(
            x=list(row.index),
            y=(row.values * 100),
            mode='lines+markers',
            name=str(cohort),
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate='Month %{x}<br>Retention: %{y:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        height=450,
        xaxis_title='Months Since Acquisition',
        yaxis_title='Retention Rate (%)',
        title='Cohort Retention Comparison Over Time',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.4,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=60, r=20, t=60, b=100)
    )
    
    return fig


def create_cohort_summary_table():
    """Create cohort summary table."""
    if cohort_summary.empty:
        return html.Div("No cohort data available")
    
    # Format the dataframe for display
    display_df = cohort_summary.copy()
    display_df['StartDate'] = display_df['StartDate'].dt.strftime('%Y-%m')
    display_df['AvgCustomerValue'] = display_df['AvgCustomerValue'].round(2)
    
    table = dbc.Table.from_dataframe(
        display_df,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="table-sm"
    )
    return table


# Create layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Customer Segmentation Dashboard",
                   className="text-center text-white mb-4",
                   style={'backgroundColor': COLORS['primary'], 'padding': '20px', 'borderRadius': '10px'})
        ])
    ], className="mb-4"),
    
    # Key Metrics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Customers", className="card-title"),
                    html.H2(f"{len(rfm_segmented):,}", className="text-primary"),
                    html.P("Unique customer IDs", className="text-muted")
                ])
            ], color="light")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Revenue", className="card-title"),
                    html.H2(f"₹{rfm_segmented['Monetary'].sum():,.0f}", className="text-success"),
                    html.P("Cumulative sales", className="text-muted")
                ])
            ], color="light")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Avg. Order Value", className="card-title"),
                    html.H2(f"₹{rfm_segmented['Monetary'].mean():,.2f}", className="text-info"),
                    html.P("Per customer", className="text-muted")
                ])
            ], color="light")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Customer Segments", className="card-title"),
                    html.H2(f"{rfm_segmented['Segment'].nunique()}", className="text-warning"),
                    html.P("Distinct groups", className="text-muted")
                ])
            ], color="light")
        ], width=3),
    ], className="mb-4"),
    
    # Main visualizations
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Customer Segment Distribution")),
                dbc.CardBody([
                    dcc.Graph(id='segment-pie-chart', figure=create_segment_pie())
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Revenue by Segment")),
                dbc.CardBody([
                    dcc.Graph(id='segment-revenue-chart', figure=create_revenue_bar())
                ])
            ])
        ], width=6),
    ], className="mb-4"),
    
    # RFM 3D Scatter
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("RFM Score Distribution (3D)")),
                dbc.CardBody([
                    dcc.Graph(id='rfm-3d-scatter', figure=create_3d_scatter())
                ])
            ])
        ])
    ], className="mb-4"),
    
    # RFM bubble plot (Recency vs Frequency, bubble = Monetary)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Recency vs Frequency (bubble = Monetary)")),
                dbc.CardBody([
                    dcc.Graph(id='rfm-bubble', figure=create_rfm_bubble())
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Segment details and metrics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Segment Summary Table")),
                dbc.CardBody([
                    html.Div(id='segment-table', children=create_segment_table())
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Time trends
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Monthly Customer & Revenue Trends")),
                dbc.CardBody([
                    dcc.Graph(id='time-trends', figure=create_time_trends())
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Cohort Analysis Section
    dbc.Row([
        dbc.Col([
            html.H2("Cohort Analysis", className="text-center mb-4",
                   style={'color': COLORS['primary'], 'padding': '15px', 'backgroundColor': '#e9ecef', 'borderRadius': '10px'})
        ])
    ], className="mb-4"),
    
    # Cohort retention heatmap
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Customer Retention by Cohort")),
                dbc.CardBody([
                    dcc.Graph(id='cohort-retention-heatmap', figure=create_cohort_retention_heatmap())
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Cohort revenue heatmap
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Revenue Retention by Cohort")),
                dbc.CardBody([
                    dcc.Graph(id='cohort-revenue-heatmap', figure=create_cohort_revenue_heatmap())
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Cohort comparison and summary
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Cohort Retention Comparison")),
                dbc.CardBody([
                    dcc.Graph(id='cohort-comparison', figure=create_cohort_comparison_chart())
                ])
            ])
        ], width=8),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Cohort Summary")),
                dbc.CardBody([
                    html.Div(id='cohort-summary-table', children=create_cohort_summary_table())
                ])
            ])
        ], width=4),
    ], className="mb-4"),
    
    # Top customers
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Top 20 High-Value Customers")),
                dbc.CardBody([
                    dcc.Graph(id='top-customers', figure=create_top_customers())
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("RFM Metrics Heatmap")),
                dbc.CardBody([
                    dcc.Graph(id='rfm-heatmap', figure=create_heatmap())
                ])
            ])
        ], width=6),
    ], className="mb-4"),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Customer Segmentation Dashboard | Powered by Dash & Plotly",
                  className="text-center text-muted")
        ])
    ])
    
], fluid=True, style={'backgroundColor': '#f8f9fa', 'padding': '20px'})


# Visualization functions
def create_segment_pie():
    """Create segment distribution pie chart."""
    segment_counts = rfm_segmented['Segment'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=segment_counts.index,
        values=segment_counts.values,
        hole=0.4,
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_revenue_bar():
    """Create revenue by segment bar chart."""
    segment_revenue = rfm_segmented.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
    
    fig = go.Figure(data=[go.Bar(
        x=segment_revenue.index,
        y=segment_revenue.values,
        marker=dict(color=segment_revenue.values, colorscale='Viridis'),
        text=[f'₹{val:,.0f}' for val in segment_revenue.values],
        textposition='outside'
    )])
    
    fig.update_layout(
        height=400,
        xaxis_title='Segment',
        yaxis_title='Total Revenue (₹)',
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    return fig


def create_3d_scatter():
    """Create 3D scatter plot of RFM scores."""
    fig = go.Figure(data=[go.Scatter3d(
        x=rfm_segmented['R_Score'],
        y=rfm_segmented['F_Score'],
        z=rfm_segmented['M_Score'],
        mode='markers',
        marker=dict(
            size=4,
            color=rfm_segmented['RFM_Value'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="RFM Value"),
            opacity=0.7
        ),
        text=rfm_segmented['Segment'],
        hovertemplate='<b>%{text}</b><br>R: %{x}<br>F: %{y}<br>M: %{z}<extra></extra>'
    )])
    
    fig.update_layout(
        height=600,
        scene=dict(
            xaxis_title='Recency Score',
            yaxis_title='Frequency Score',
            zaxis_title='Monetary Score'
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


def create_segment_table():
    """Create segment summary table."""
    table = dbc.Table.from_dataframe(
        segment_summary,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="table-sm"
    )
    return table


def create_time_trends():
    """Create time trends chart."""
    cleaned_df['YearMonth'] = cleaned_df['InvoiceDate'].dt.to_period('M').astype(str)
    monthly_data = cleaned_df.groupby('YearMonth').agg({
        'CustomerID': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=monthly_data['YearMonth'], y=monthly_data['CustomerID'],
                  name='Customers', line=dict(color='blue', width=2)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=monthly_data['YearMonth'], y=monthly_data['TotalAmount'],
                  name='Revenue', line=dict(color='green', width=2)),
        secondary_y=True
    )
    
    fig.update_layout(
        height=400,
        hovermode='x unified',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    fig.update_xaxes(title_text='Month', tickangle=45)
    fig.update_yaxes(title_text='Customers', secondary_y=False)
    fig.update_yaxes(title_text='Revenue ($)', secondary_y=True)
    
    return fig


def create_top_customers():
    """Create top customers chart."""
    top_20 = rfm_segmented.nlargest(20, 'Monetary')
    
    fig = go.Figure(data=[go.Bar(
        x=top_20['CustomerID'].astype(str),
        y=top_20['Monetary'],
        marker=dict(color=top_20['Monetary'], colorscale='Blues'),
        text=top_20['Segment'],
        hovertemplate='Customer: %{x}<br>Revenue: ₹%{y:,.0f}<br>Segment: %{text}<extra></extra>'
    )])
    
    fig.update_layout(
        height=400,
        xaxis_title='Customer ID',
        yaxis_title='Total Spending ($)',
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    fig.update_xaxes(tickangle=45)
    
    return fig


def create_heatmap():
    """Create RFM correlation heatmap."""
    corr_data = rfm_segmented[['Recency', 'Frequency', 'Monetary']].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns,
        y=corr_data.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_data.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 14}
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_cohort_retention_heatmap():
    """Create cohort retention heatmap."""
    if cohort_retention.empty:
        return go.Figure()
    
    # Convert to percentages and round
    retention_pct = (cohort_retention * 100).round(1)
    
    fig = go.Figure(data=go.Heatmap(
        z=retention_pct.values,
        x=[f'Month {i}' for i in retention_pct.columns],
        y=[str(idx) for idx in retention_pct.index],
        colorscale='YlOrRd',
        text=retention_pct.values,
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        hovertemplate='Cohort: %{y}<br>Month: %{x}<br>Retention: %{z:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        height=500,
        xaxis_title='Period (Months)',
        yaxis_title='Acquisition Cohort',
        title='Customer Retention Rate by Cohort (%)',
        margin=dict(l=80, r=20, t=60, b=40)
    )
    
    return fig


def create_cohort_revenue_heatmap():
    """Create cohort revenue retention heatmap."""
    if cohort_revenue.empty:
        return go.Figure()
    
    # Convert to percentages and round
    revenue_pct = (cohort_revenue * 100).round(1)
    
    fig = go.Figure(data=go.Heatmap(
        z=revenue_pct.values,
        x=[f'Month {i}' for i in revenue_pct.columns],
        y=[str(idx) for idx in revenue_pct.index],
        colorscale='Blues',
        text=revenue_pct.values,
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        hovertemplate='Cohort: %{y}<br>Month: %{x}<br>Revenue Retention: %{z:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        height=500,
        xaxis_title='Period (Months)',
        yaxis_title='Acquisition Cohort',
        title='Revenue Retention Rate by Cohort (%)',
        margin=dict(l=80, r=20, t=60, b=40)
    )
    
    return fig


def create_cohort_comparison_chart():
    """Create cohort comparison line chart."""
    if cohort_retention.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Plot retention curves for each cohort
    colors = px.colors.qualitative.Set3
    
    for i, (cohort, row) in enumerate(cohort_retention.iterrows()):
        fig.add_trace(go.Scatter(
            x=list(row.index),
            y=(row.values * 100),
            mode='lines+markers',
            name=str(cohort),
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate='Month %{x}<br>Retention: %{y:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        height=450,
        xaxis_title='Months Since Acquisition',
        yaxis_title='Retention Rate (%)',
        title='Cohort Retention Comparison Over Time',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.4,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=60, r=20, t=60, b=100)
    )
    
    return fig


def create_cohort_summary_table():
    """Create cohort summary table."""
    if cohort_summary.empty:
        return html.Div("No cohort data available")
    
    # Format the dataframe for display
    display_df = cohort_summary.copy()
    display_df['StartDate'] = display_df['StartDate'].dt.strftime('%Y-%m')
    display_df['AvgCustomerValue'] = display_df['AvgCustomerValue'].round(2)
    
    table = dbc.Table.from_dataframe(
        display_df,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="table-sm"
    )
    return table


# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    print("\n" + "="*80)
    print("STARTING DASHBOARD SERVER")
    print("="*80)
    print(f"\n📊 Dashboard is running at: http://0.0.0.0:{port}")
    print("   Press Ctrl+C to stop the server\n")

    app.run_server(debug=False, host='0.0.0.0', port=port)
