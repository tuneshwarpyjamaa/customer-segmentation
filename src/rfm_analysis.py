"""
RFM (Recency, Frequency, Monetary) Analysis Module

Implements customer segmentation based on RFM methodology.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class RFMAnalyzer:
    """
    RFM Analysis implementation for customer segmentation.
    """
    
    def __init__(self, df: pd.DataFrame, customer_col: str = 'CustomerID',
                 date_col: str = 'InvoiceDate', amount_col: str = 'TotalAmount',
                 invoice_col: str = 'InvoiceNo'):
        """
        Initialize RFM Analyzer.

        Args:
            df: Cleaned transaction DataFrame
            customer_col: Customer identifier column
            date_col: Transaction date column
            amount_col: Transaction amount column
            invoice_col: Invoice identifier column
        """
        self.df = df
        self.customer_col = customer_col
        self.date_col = date_col
        self.amount_col = amount_col
        self.invoice_col = invoice_col

        # Reference date for recency calculation (last date in dataset + 1 day)
        if not df.empty and date_col in df.columns:
            self.reference_date = df[date_col].max() + timedelta(days=1)
        else:
            self.reference_date = None
        
    def calculate_rfm_metrics(self) -> pd.DataFrame:
        """
        Calculate RFM metrics for each customer.
        
        Returns:
            DataFrame with RFM metrics
        """
        print("Calculating RFM metrics...")
        
        # Aggregate data by customer
        rfm = self.df.groupby(self.customer_col).agg({
            self.date_col: lambda x: (self.reference_date - x.max()).days,  # Recency
            self.invoice_col: 'nunique',  # Frequency
            self.amount_col: 'sum'  # Monetary
        }).reset_index()
        
        # Rename columns
        rfm.columns = [self.customer_col, 'Recency', 'Frequency', 'Monetary']
        
        # Convert to appropriate data types for memory efficiency
        rfm['Recency'] = rfm['Recency'].astype('uint16')
        rfm['Frequency'] = rfm['Frequency'].astype('uint16')
        rfm['Monetary'] = rfm['Monetary'].astype('float32')
        
        print(f"  - Calculated RFM for {len(rfm):,} customers")
        
        return rfm
    
    def assign_rfm_scores(self, rfm: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
        """
        Assign RFM scores (1-5) based on quantiles.
        
        Args:
            rfm: DataFrame with RFM metrics
            n_quantiles: Number of quantiles (default 5 for quintiles)
            
        Returns:
            DataFrame with RFM scores added
        """
        print(f"Assigning RFM scores ({n_quantiles} quantiles)...")
        
        # Create scores based on quantiles
        # For Recency: Lower is better (more recent), so we reverse the score
        rfm['R_Score'] = pd.qcut(rfm['Recency'], q=n_quantiles, labels=range(n_quantiles, 0, -1), duplicates='drop')
        
        # For Frequency: Higher is better
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=n_quantiles, labels=range(1, n_quantiles+1), duplicates='drop')
        
        # For Monetary: Higher is better
        rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=n_quantiles, labels=range(1, n_quantiles+1), duplicates='drop')
        
        # Convert scores to numeric for calculations
        rfm['R_Score'] = rfm['R_Score'].astype('uint8')
        rfm['F_Score'] = rfm['F_Score'].astype('uint8')
        rfm['M_Score'] = rfm['M_Score'].astype('uint8')
        
        # Create combined RFM score
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        # Calculate overall RFM value (average of R, F, M scores)
        rfm['RFM_Value'] = (rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']) / 3
        rfm['RFM_Value'] = rfm['RFM_Value'].astype('float32')
        
        print("  - Assigned R, F, M scores")
        
        return rfm
    
    def create_segments(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer segments based on RFM scores.
        
        Args:
            rfm: DataFrame with RFM scores
            
        Returns:
            DataFrame with segments added
        """
        print("Creating customer segments...")
        
        def segment_customer(row):
            """Assign segment based on RFM scores."""
            r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
            
            # Champions: Best customers
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            
            # Loyal Customers: Regular purchasers
            elif r >= 3 and f >= 3 and m >= 3:
                return 'Loyal'
            
            # Potential Loyalists: Recent customers with potential
            elif r >= 3 and f >= 1 and m >= 1:
                return 'Potential Loyalist'
            
            # New Customers: Very recent but low frequency
            elif r >= 4 and f <= 2 and m <= 2:
                return 'New Customers'
            
            # Promising: Recent with moderate frequency
            elif r >= 3 and f >= 2 and m >= 2:
                return 'Promising'
            
            # Need Attention: Average on all scores
            elif r >= 2 and f >= 2 and m >= 2:
                return 'Need Attention'
            
            # About to Sleep: Below average recency
            elif r >= 2 and f <= 2 and m <= 2:
                return 'About to Sleep'
            
            # At Risk: Used to be good, need re-engagement
            elif r <= 2 and f >= 3 and m >= 3:
                return 'At Risk'
            
            # Cannot Lose Them: High value but haven't purchased recently
            elif r <= 2 and f >= 4 and m >= 4:
                return 'Cannot Lose'
            
            # Hibernating: Low engagement across the board
            elif r <= 2 and f <= 2 and m <= 2:
                return 'Hibernating'
            
            # Lost: Very low on all metrics
            else:
                return 'Lost'
        
        rfm['Segment'] = rfm.apply(segment_customer, axis=1)
        # Only convert to categories that actually exist in the data
        rfm['Segment'] = pd.Categorical(rfm['Segment'], categories=rfm['Segment'].unique())
        
        # Count customers in each segment
        segment_counts = rfm['Segment'].value_counts().sort_values(ascending=False)
        print("\n  Segment Distribution:")
        for segment, count in segment_counts.items():
            percentage = (count / len(rfm)) * 100
            print(f"    {segment:.<25} {count:>6,} ({percentage:>5.2f}%)")
        
        return rfm


def calculate_rfm(df: pd.DataFrame, customer_col: str = 'CustomerID',
                  date_col: str = 'InvoiceDate', amount_col: str = 'TotalAmount',
                  invoice_col: str = 'InvoiceNo') -> pd.DataFrame:
    """
    Main function to calculate RFM metrics.
    
    Args:
        df: Cleaned transaction DataFrame
        customer_col: Customer identifier column
        date_col: Transaction date column
        amount_col: Transaction amount column
        invoice_col: Invoice identifier column
        
    Returns:
        DataFrame with RFM metrics
    """
    analyzer = RFMAnalyzer(df, customer_col, date_col, amount_col, invoice_col)
    rfm = analyzer.calculate_rfm_metrics()
    
    return rfm


def segment_customers(rfm: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """
    Segment customers based on RFM scores.
    
    Args:
        rfm: DataFrame with RFM metrics
        n_quantiles: Number of quantiles for scoring
        
    Returns:
        DataFrame with segments
    """
    # Create a dummy analyzer instance for segmentation
    analyzer = RFMAnalyzer(pd.DataFrame())
    
    # Assign scores
    rfm = analyzer.assign_rfm_scores(rfm, n_quantiles)
    
    # Create segments
    rfm = analyzer.create_segments(rfm)
    
    return rfm


def get_segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for each segment.
    
    Args:
        rfm: DataFrame with RFM scores and segments
        
    Returns:
        Summary DataFrame
    """
    summary = rfm.groupby('Segment').agg({
        'CustomerID': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'sum'],
        'RFM_Value': 'mean'
    }).round(2)
    
    # Flatten column names
    summary.columns = ['Customer_Count', 'Avg_Recency', 'Avg_Frequency', 
                       'Avg_Monetary', 'Total_Revenue', 'Avg_RFM_Value']
    
    # Calculate percentage of total customers and revenue
    summary['Customer_Percentage'] = (summary['Customer_Count'] / summary['Customer_Count'].sum() * 100).round(2)
    summary['Revenue_Percentage'] = (summary['Total_Revenue'] / summary['Total_Revenue'].sum() * 100).round(2)
    
    # Sort by Total_Revenue descending
    summary = summary.sort_values('Total_Revenue', ascending=False)
    
    return summary.reset_index()


def identify_high_value_customers(rfm: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """
    Identify top N high-value customers.
    
    Args:
        rfm: DataFrame with RFM scores
        top_n: Number of top customers to return
        
    Returns:
        DataFrame with top customers
    """
    # Sort by Monetary value and get top N
    top_customers = rfm.nlargest(top_n, 'Monetary')[
        ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment', 'RFM_Score']
    ]
    
    return top_customers


def calculate_customer_lifetime_value(rfm: pd.DataFrame, time_period_days: int = 365) -> pd.DataFrame:
    """
    Estimate Customer Lifetime Value (CLV).
    
    Args:
        rfm: DataFrame with RFM metrics
        time_period_days: Time period for CLV calculation (default 365 days)
        
    Returns:
        DataFrame with CLV added
    """
    # Simple CLV calculation: (Average Order Value) × (Purchase Frequency) × (Customer Lifespan)
    rfm['AOV'] = rfm['Monetary'] / rfm['Frequency']
    rfm['Purchase_Frequency'] = rfm['Frequency'] / (time_period_days / 365)
    rfm['CLV_Estimate'] = rfm['AOV'] * rfm['Purchase_Frequency'] * 3  # Assuming 3-year lifespan
    rfm['CLV_Estimate'] = rfm['CLV_Estimate'].astype('float32')
    
    return rfm


if __name__ == "__main__":
    # Test RFM analysis
    from pathlib import Path
    from data_loader import load_data_optimized
    from data_cleaner import clean_data
    
    data_path = Path(__file__).parent.parent / "data" / "raw" / "OnlineRetail.xlsx"
    
    if data_path.exists():
        print("\n" + "="*80)
        print("RFM ANALYSIS TEST")
        print("="*80 + "\n")
        
        # Load and clean data
        df = load_data_optimized(str(data_path))
        cleaned_df = clean_data(df)
        
        # Calculate RFM
        rfm = calculate_rfm(cleaned_df)
        rfm_segmented = segment_customers(rfm)
        
        # Get summary
        print("\n" + "="*80)
        print("SEGMENT SUMMARY")
        print("="*80)
        print(get_segment_summary(rfm_segmented))
        
        # High-value customers
        print("\n" + "="*80)
        print("TOP 10 HIGH-VALUE CUSTOMERS")
        print("="*80)
        print(identify_high_value_customers(rfm_segmented, top_n=10))
        
    else:
        print(f"Data file not found at {data_path}")
