"""
Data Cleaning and Preprocessing Module

Handles missing values, outliers, and data quality issues.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """
    Data cleaning and preprocessing utilities.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataCleaner with a DataFrame.
        
        Args:
            df: Input DataFrame
        """
        self.df = df.copy()
        self.original_shape = df.shape
        
    def handle_missing_values(self) -> 'DataCleaner':
        """
        Handle missing values in the dataset.
        
        Returns:
            Self for method chaining
        """
        print("Handling missing values...")
        
        # Remove rows where CustomerID is missing (can't segment without it)
        if 'CustomerID' in self.df.columns:
            before = len(self.df)
            self.df = self.df[self.df['CustomerID'].notna()]
            removed = before - len(self.df)
            print(f"  - Removed {removed:,} rows with missing CustomerID")
        
        # Remove rows where Description is missing
        if 'Description' in self.df.columns:
            before = len(self.df)
            self.df = self.df[self.df['Description'].notna()]
            removed = before - len(self.df)
            print(f"  - Removed {removed:,} rows with missing Description")
        
        return self
    
    def remove_invalid_quantities(self) -> 'DataCleaner':
        """
        Remove records with invalid (negative or zero) quantities.
        
        Returns:
            Self for method chaining
        """
        print("Removing invalid quantities...")
        
        if 'Quantity' in self.df.columns:
            before = len(self.df)
            self.df = self.df[self.df['Quantity'] > 0]
            removed = before - len(self.df)
            print(f"  - Removed {removed:,} rows with invalid quantities")
        
        return self
    
    def remove_invalid_prices(self) -> 'DataCleaner':
        """
        Remove records with invalid (negative or zero) prices.
        
        Returns:
            Self for method chaining
        """
        print("Removing invalid prices...")
        
        if 'UnitPrice' in self.df.columns:
            before = len(self.df)
            self.df = self.df[self.df['UnitPrice'] > 0]
            removed = before - len(self.df)
            print(f"  - Removed {removed:,} rows with invalid prices")
        
        return self
    
    def remove_cancelled_orders(self) -> 'DataCleaner':
        """
        Remove cancelled orders (InvoiceNo starting with 'C').
        
        Returns:
            Self for method chaining
        """
        print("Removing cancelled orders...")
        
        if 'InvoiceNo' in self.df.columns:
            before = len(self.df)
            self.df = self.df[~self.df['InvoiceNo'].astype(str).str.startswith('C')]
            removed = before - len(self.df)
            print(f"  - Removed {removed:,} cancelled orders")
        
        return self
    
    def handle_outliers_iqr(self, column: str, multiplier: float = 1.5) -> 'DataCleaner':
        """
        Remove outliers using IQR method.
        
        Args:
            column: Column name to check for outliers
            multiplier: IQR multiplier (default 1.5 for standard outlier detection)
            
        Returns:
            Self for method chaining
        """
        print(f"Handling outliers in {column}...")
        
        if column in self.df.columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            before = len(self.df)
            self.df = self.df[
                (self.df[column] >= lower_bound) & 
                (self.df[column] <= upper_bound)
            ]
            removed = before - len(self.df)
            print(f"  - Removed {removed:,} outliers from {column}")
            print(f"  - Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return self
    
    def convert_dates(self, date_column: str = 'InvoiceDate') -> 'DataCleaner':
        """
        Convert date columns to datetime format.
        
        Args:
            date_column: Name of the date column
            
        Returns:
            Self for method chaining
        """
        print("Converting date columns...")
        
        if date_column in self.df.columns:
            self.df[date_column] = pd.to_datetime(self.df[date_column])
            print(f"  - Converted {date_column} to datetime")
        
        return self
    
    def create_derived_features(self) -> 'DataCleaner':
        """
        Create useful derived features.
        
        Returns:
            Self for method chaining
        """
        print("Creating derived features...")
        
        # Total amount per transaction
        if 'Quantity' in self.df.columns and 'UnitPrice' in self.df.columns:
            self.df['TotalAmount'] = self.df['Quantity'] * self.df['UnitPrice']
            print("  - Created TotalAmount feature")
        
        # Extract date components
        if 'InvoiceDate' in self.df.columns:
            self.df['Year'] = self.df['InvoiceDate'].dt.year.astype('uint16')
            self.df['Month'] = self.df['InvoiceDate'].dt.month.astype('uint8')
            self.df['Day'] = self.df['InvoiceDate'].dt.day.astype('uint8')
            self.df['DayOfWeek'] = self.df['InvoiceDate'].dt.dayofweek.astype('uint8')
            self.df['Hour'] = self.df['InvoiceDate'].dt.hour.astype('uint8')
            print("  - Created date component features")
        
        return self
    
    def remove_duplicates(self) -> 'DataCleaner':
        """
        Remove duplicate records.
        
        Returns:
            Self for method chaining
        """
        print("Removing duplicates...")
        
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = before - len(self.df)
        print(f"  - Removed {removed:,} duplicate rows")
        
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Get the cleaned DataFrame.
        
        Returns:
            Cleaned DataFrame
        """
        return self.df
    
    def print_cleaning_summary(self) -> None:
        """
        Print summary of cleaning operations.
        """
        print("\n" + "="*80)
        print("CLEANING SUMMARY")
        print("="*80)
        print(f"Original shape: {self.original_shape[0]:,} rows × {self.original_shape[1]} columns")
        print(f"Cleaned shape:  {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"Rows removed:   {self.original_shape[0] - self.df.shape[0]:,} ({(1 - self.df.shape[0]/self.original_shape[0])*100:.2f}%)")
        print(f"Data quality:   {(self.df.shape[0]/self.original_shape[0])*100:.2f}% retained")
        print("="*80 + "\n")


def clean_data(df: pd.DataFrame, remove_outliers: bool = True) -> pd.DataFrame:
    """
    Main function to clean the retail dataset.
    
    Args:
        df: Input DataFrame
        remove_outliers: Whether to remove outliers (default True)
        
    Returns:
        Cleaned DataFrame
    """
    print("\n" + "="*80)
    print("STARTING DATA CLEANING")
    print("="*80 + "\n")
    
    cleaner = DataCleaner(df)
    
    # Apply cleaning steps
    cleaner.convert_dates() \
           .handle_missing_values() \
           .remove_cancelled_orders() \
           .remove_invalid_quantities() \
           .remove_invalid_prices() \
           .remove_duplicates() \
           .create_derived_features()
    
    # Optional: Remove outliers
    if remove_outliers:
        cleaner.handle_outliers_iqr('Quantity', multiplier=3.0) \
               .handle_outliers_iqr('UnitPrice', multiplier=3.0) \
               .handle_outliers_iqr('TotalAmount', multiplier=3.0)
    
    # Print summary
    cleaner.print_cleaning_summary()
    
    return cleaner.get_cleaned_data()


def get_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a data quality report.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with quality metrics
    """
    report = pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': [df[col].notna().sum() for col in df.columns],
        'Null Count': [df[col].isna().sum() for col in df.columns],
        'Null %': [df[col].isna().sum() / len(df) * 100 for col in df.columns],
        'Unique Values': [df[col].nunique() for col in df.columns],
        'Data Type': [df[col].dtype for col in df.columns]
    })
    
    return report


if __name__ == "__main__":
    # Test the cleaner
    from pathlib import Path
    from data_loader import load_data_optimized
    
    data_path = Path(__file__).parent.parent / "data" / "raw" / "OnlineRetail.xlsx"
    
    if data_path.exists():
        df = load_data_optimized(str(data_path))
        cleaned_df = clean_data(df)
        
        print("\nData Quality Report:")
        print(get_data_quality_report(cleaned_df))
    else:
        print(f"Data file not found at {data_path}")
