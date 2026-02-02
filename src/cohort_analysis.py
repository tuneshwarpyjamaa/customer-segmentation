"""
Cohort Analysis Module for Customer Segmentation

Tracks customer retention and value over time based on acquisition cohorts.
"""

from operator import attrgetter
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CohortAnalyzer:
    """
    Cohort Analysis implementation for tracking customer retention and value.
    """
    
    def __init__(self, df: pd.DataFrame, customer_col: str = 'CustomerID',
                 date_col: str = 'InvoiceDate', amount_col: str = 'TotalAmount'):
        """
        Initialize Cohort Analyzer.

        Args:
            df: Cleaned transaction DataFrame
            customer_col: Customer identifier column
            date_col: Transaction date column
            amount_col: Transaction amount column
        """
        self.df = df.copy()
        self.customer_col = customer_col
        self.date_col = date_col
        self.amount_col = amount_col
        
        # Ensure date column is datetime
        self.df[date_col] = pd.to_datetime(self.df[date_col])
    
    def get_customer_acquisition_date(self) -> pd.DataFrame:
        """
        Get the first purchase date (acquisition date) for each customer.
        
        Returns:
            DataFrame with customer ID and acquisition date
        """
        acquisition_df = self.df.groupby(self.customer_col)[self.date_col].min().reset_index()
        acquisition_df.columns = [self.customer_col, 'AcquisitionDate']
        return acquisition_df
    
    def create_cohort_table(self, period: str = 'M') -> pd.DataFrame:
        """
        Create cohort table showing retention rates over time.
        
        Args:
            period: Time period for cohorts ('M' for month, 'W' for week, 'Q' for quarter)
            
        Returns:
            Cohort table DataFrame with retention percentages
        """
        # Get acquisition dates
        acquisition_df = self.get_customer_acquisition_date()
        
        # Merge back to main dataframe
        df_with_acquisition = self.df.merge(acquisition_df, on=self.customer_col)
        
        # Create period labels
        period_map = {'M': 'M', 'W': 'W', 'Q': 'Q'}
        period_label = period_map.get(period, 'M')
        
        # Get cohort group (acquisition period)
        df_with_acquisition['CohortGroup'] = df_with_acquisition['AcquisitionDate'].dt.to_period(period_label)
        
        # Get activity period
        df_with_acquisition['ActivityPeriod'] = df_with_acquisition[self.date_col].dt.to_period(period_label)
        
        # Calculate period number (difference between activity and acquisition)
        df_with_acquisition['PeriodNumber'] = (
            df_with_acquisition['ActivityPeriod'] - df_with_acquisition['CohortGroup']
        ).apply(attrgetter('n'))
        
        # Group by cohort and period to get unique active customers
        cohort_data = df_with_acquisition.groupby(['CohortGroup', 'PeriodNumber'])[self.customer_col].nunique().reset_index()
        cohort_data.columns = ['CohortGroup', 'PeriodNumber', 'ActiveCustomers']
        
        # Get initial cohort sizes
        cohort_sizes = acquisition_df.groupby(
            acquisition_df['AcquisitionDate'].dt.to_period(period_label)
        )[self.customer_col].nunique().reset_index()
        cohort_sizes.columns = ['CohortGroup', 'CohortSize']
        
        # Merge and calculate retention rate
        cohort_table = cohort_data.merge(cohort_sizes, on='CohortGroup')
        cohort_table['RetentionRate'] = cohort_table['ActiveCustomers'] / cohort_table['CohortSize']
        
        # Pivot to create cohort matrix
        cohort_matrix = cohort_table.pivot_table(
            index='CohortGroup',
            columns='PeriodNumber',
            values='RetentionRate',
            fill_value=0
        )
        # Ensure the index (CohortGroup) is JSON-serializable (no pandas Period objects)
        try:
            cohort_matrix.index = cohort_matrix.index.astype(str)
        except Exception:
            pass

        return cohort_matrix
    
    def create_cohort_revenue_table(self, period: str = 'M') -> pd.DataFrame:
        """
        Create cohort table showing revenue retention over time.
        
        Args:
            period: Time period for cohorts ('M' for month, 'W' for week, 'Q' for quarter)
            
        Returns:
            Cohort revenue table DataFrame
        """
        # Get acquisition dates
        acquisition_df = self.get_customer_acquisition_date()
        
        # Merge back to main dataframe
        df_with_acquisition = self.df.merge(acquisition_df, on=self.customer_col)
        
        # Create period labels
        period_label = period
        
        # Get cohort group (acquisition period)
        df_with_acquisition['CohortGroup'] = df_with_acquisition['AcquisitionDate'].dt.to_period(period_label)
        
        # Get activity period
        df_with_acquisition['ActivityPeriod'] = df_with_acquisition[self.date_col].dt.to_period(period_label)
        
        # Calculate period number
        df_with_acquisition['PeriodNumber'] = (
            df_with_acquisition['ActivityPeriod'] - df_with_acquisition['CohortGroup']
        ).apply(attrgetter('n'))
        
        # Group by cohort and period to get revenue
        cohort_revenue = df_with_acquisition.groupby(['CohortGroup', 'PeriodNumber'])[self.amount_col].sum().reset_index()
        cohort_revenue.columns = ['CohortGroup', 'PeriodNumber', 'Revenue']
        
        # Get initial cohort revenue (first period)
        initial_revenue = cohort_revenue[cohort_revenue['PeriodNumber'] == 0].copy()
        initial_revenue.columns = ['CohortGroup', 'PeriodNumber', 'InitialRevenue']
        initial_revenue = initial_revenue[['CohortGroup', 'InitialRevenue']]
        
        # Merge and calculate revenue retention rate
        cohort_revenue_table = cohort_revenue.merge(initial_revenue, on='CohortGroup')
        cohort_revenue_table['RevenueRetention'] = cohort_revenue_table['Revenue'] / cohort_revenue_table['InitialRevenue']
        
        # Pivot to create cohort matrix
        cohort_revenue_matrix = cohort_revenue_table.pivot_table(
            index='CohortGroup',
            columns='PeriodNumber',
            values='RevenueRetention',
            fill_value=0
        )
        # Convert Period index to string for serialization
        try:
            cohort_revenue_matrix.index = cohort_revenue_matrix.index.astype(str)
        except Exception:
            pass

        return cohort_revenue_matrix
    
    def get_cohort_summary(self, period: str = 'M') -> pd.DataFrame:
        """
        Get summary statistics for each cohort.
        
        Args:
            period: Time period for cohorts
            
        Returns:
            Summary DataFrame with cohort metrics
        """
        # Get acquisition dates
        acquisition_df = self.get_customer_acquisition_date()
        period_label = period
        
        # Add cohort period column
        acquisition_df['CohortPeriod'] = acquisition_df['AcquisitionDate'].dt.to_period(period_label)
        
        # Group by cohort
        cohort_stats = acquisition_df.groupby('CohortPeriod').agg({
            self.customer_col: 'count',
            'AcquisitionDate': 'first'
        }).reset_index()
        cohort_stats.columns = ['CohortGroup', 'CustomerCount', 'StartDate']
        
        # Calculate average customer value for each cohort
        customer_values = self.df.groupby(self.customer_col)[self.amount_col].sum().reset_index()
        customer_values.columns = [self.customer_col, 'TotalValue']
        
        cohort_customer_values = acquisition_df.merge(customer_values, on=self.customer_col)
        cohort_avg_values = cohort_customer_values.groupby(
            cohort_customer_values['AcquisitionDate'].dt.to_period(period_label)
        )['TotalValue'].mean().reset_index()
        cohort_avg_values.columns = ['CohortGroup', 'AvgCustomerValue']
        
        # Merge stats
        cohort_stats = cohort_stats.merge(cohort_avg_values, on='CohortGroup')

        # Ensure CohortGroup is serializable
        try:
            cohort_stats['CohortGroup'] = cohort_stats['CohortGroup'].astype(str)
        except Exception:
            pass

        return cohort_stats
    
    def calculate_customer_lifetime_by_cohort(self, period: str = 'M') -> pd.DataFrame:
        """
        Calculate average customer lifetime metrics by cohort.
        
        Args:
            period: Time period for cohorts
            
        Returns:
            DataFrame with lifetime metrics per cohort
        """
        # Get acquisition dates
        acquisition_df = self.get_customer_acquisition_date()
        
        # Get last purchase date for each customer
        last_purchase = self.df.groupby(self.customer_col)[self.date_col].max().reset_index()
        last_purchase.columns = [self.customer_col, 'LastPurchaseDate']
        
        # Merge with acquisition
        customer_lifetime = acquisition_df.merge(last_purchase, on=self.customer_col)
        customer_lifetime['LifetimeDays'] = (
            customer_lifetime['LastPurchaseDate'] - customer_lifetime['AcquisitionDate']
        ).dt.days
        
        # Get total purchases and revenue per customer
        customer_stats = self.df.groupby(self.customer_col).agg({
            self.amount_col: 'sum',
            self.date_col: 'count'
        }).reset_index()
        customer_stats.columns = [self.customer_col, 'TotalRevenue', 'TotalPurchases']
        
        customer_lifetime = customer_lifetime.merge(customer_stats, on=self.customer_col)
        
        # Group by cohort
        period_label = period
        cohort_lifetime = customer_lifetime.groupby(
            customer_lifetime['AcquisitionDate'].dt.to_period(period_label)
        ).agg({
            'LifetimeDays': 'mean',
            'TotalRevenue': 'mean',
            'TotalPurchases': 'mean',
            self.customer_col: 'count'
        }).reset_index()
        cohort_lifetime.columns = ['CohortGroup', 'AvgLifetimeDays', 'AvgRevenue', 'AvgPurchases', 'CustomerCount']
        # Convert CohortGroup to string to avoid Period objects in output
        try:
            cohort_lifetime['CohortGroup'] = cohort_lifetime['CohortGroup'].astype(str)
        except Exception:
            pass

        return cohort_lifetime.round(2)


def create_cohort_analysis(df: pd.DataFrame, period: str = 'M') -> Dict[str, pd.DataFrame]:
    """
    Main function to create complete cohort analysis.
    
    Args:
        df: Cleaned transaction DataFrame
        period: Time period for cohorts ('M', 'W', 'Q')
        
    Returns:
        Dictionary containing cohort retention, revenue retention, and summary tables
    """
    print("Creating cohort analysis...")
    analyzer = CohortAnalyzer(df)
    
    # Create cohort retention table
    retention_matrix = analyzer.create_cohort_table(period)
    print(f"  - Created retention matrix: {retention_matrix.shape}")
    
    # Create cohort revenue table
    revenue_matrix = analyzer.create_cohort_revenue_table(period)
    print(f"  - Created revenue matrix: {revenue_matrix.shape}")
    
    # Get cohort summary
    cohort_summary = analyzer.get_cohort_summary(period)
    print(f"  - Created cohort summary: {len(cohort_summary)} cohorts")
    
    # Get lifetime metrics
    lifetime_metrics = analyzer.calculate_customer_lifetime_by_cohort(period)
    print(f"  - Created lifetime metrics")
    
    return {
        'retention_matrix': retention_matrix,
        'revenue_matrix': revenue_matrix,
        'cohort_summary': cohort_summary,
        'lifetime_metrics': lifetime_metrics
    }


if __name__ == "__main__":
    # Test cohort analysis
    from pathlib import Path
    from data_loader import load_data_optimized
    from data_cleaner import clean_data
    
    data_path = Path(__file__).parent.parent / "data" / "raw" / "OnlineRetail.csv"
    
    if data_path.exists():
        print("\n" + "="*80)
        print("COHORT ANALYSIS TEST")
        print("="*80 + "\n")
        
        # Load and clean data
        df = load_data_optimized(str(data_path), file_format='csv')
        cleaned_df = clean_data(df)
        
        # Create cohort analysis
        results = create_cohort_analysis(cleaned_df)
        
        print("\n" + "="*80)
        print("COHORT RETENTION MATRIX")
        print("="*80)
        print(results['retention_matrix'])
        
        print("\n" + "="*80)
        print("COHORT SUMMARY")
        print("="*80)
        print(results['cohort_summary'])
        
        print("\n" + "="*80)
        print("LIFETIME METRICS BY COHORT")
        print("="*80)
        print(results['lifetime_metrics'])
    else:
        print(f"Data file not found at {data_path}")