"""
Unit Tests for Data Cleaner Module

Tests the data cleaning and preprocessing functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_cleaner import DataCleaner, clean_data, get_data_quality_report


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing."""
    np.random.seed(42)
    
    n_rows = 500
    data = {
        'InvoiceNo': [f'INV{i:06d}' for i in range(n_rows)],
        'CustomerID': [f'C{np.random.randint(1, 50):04d}' for _ in range(n_rows)],
        'InvoiceDate': pd.date_range(start='2023-01-01', periods=n_rows, freq='H'),
        'Quantity': np.random.randint(-50, 100, n_rows),  # Include negatives for testing
        'UnitPrice': np.random.uniform(0, 100, n_rows).round(2),
        'Description': [f'Product {i}' for i in np.random.randint(1, 80, n_rows)],
        'Country': np.random.choice(['UK', 'USA', 'Germany', 'France'], n_rows),
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values and issues
    df.loc[0:9, 'CustomerID'] = np.nan
    df.loc[10:19, 'Description'] = np.nan
    df.loc[20:29, 'Quantity'] = -999  # Invalid quantity
    df.loc[30:39, 'UnitPrice'] = 0  # Invalid price
    df.loc[40:49, 'InvoiceNo'] = f'C{50:06d}'  # Cancelled orders
    
    return df


@pytest.fixture
def clean_sample_data():
    """Create a clean sample DataFrame without issues."""
    np.random.seed(42)
    
    n_rows = 100
    dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='H')
    
    data = {
        'InvoiceNo': [f'INV{i:06d}' for i in range(n_rows)],
        'CustomerID': [f'C{np.random.randint(1, 20):04d}' for _ in range(n_rows)],
        'InvoiceDate': dates,
        'Quantity': np.random.randint(1, 50, n_rows),
        'UnitPrice': np.random.uniform(1, 100, n_rows).round(2),
        'Description': [f'Product {i}' for i in np.random.randint(1, 50, n_rows)],
    }
    
    return pd.DataFrame(data)


class TestDataCleaner:
    """Test cases for DataCleaner class."""
    
    def test_init_creates_copy(self, sample_transaction_data):
        """Test that initialization creates a copy of the data."""
        cleaner = DataCleaner(sample_transaction_data)
        
        assert cleaner.df is not sample_transaction_data
        assert cleaner.original_shape == sample_transaction_data.shape
    
    def test_init_stores_original_shape(self, sample_transaction_data):
        """Test that original shape is stored."""
        cleaner = DataCleaner(sample_transaction_data)
        
        assert cleaner.original_shape == sample_transaction_data.shape
    
    def test_handle_missing_values(self, sample_transaction_data):
        """Test handling of missing values."""
        initial_count = len(sample_transaction_data)
        cleaner = DataCleaner(sample_transaction_data)
        cleaner.handle_missing_values()
        
        # Should have removed rows with missing CustomerID and Description
        assert cleaner.df['CustomerID'].notna().all()
        assert cleaner.df['Description'].notna().all()
        assert len(cleaner.df) < initial_count
    
    def test_handle_missing_values_no_customerid_column(self):
        """Test handling missing values when CustomerID column doesn't exist."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        cleaner = DataCleaner(df)
        result = cleaner.handle_missing_values()
        
        assert result is not None
    
    def test_remove_invalid_quantities(self, sample_transaction_data):
        """Test removal of invalid (negative/zero) quantities."""
        initial_count = len(sample_transaction_data)
        cleaner = DataCleaner(sample_transaction_data)
        cleaner.remove_invalid_quantities()
        
        assert (cleaner.df['Quantity'] > 0).all()
        assert len(cleaner.df) < initial_count
    
    def test_remove_invalid_prices(self, sample_transaction_data):
        """Test removal of invalid (negative/zero) prices."""
        initial_count = len(sample_transaction_data)
        cleaner = DataCleaner(sample_transaction_data)
        cleaner.remove_invalid_prices()
        
        assert (cleaner.df['UnitPrice'] > 0).all()
        assert len(cleaner.df) < initial_count
    
    def test_remove_cancelled_orders(self, sample_transaction_data):
        """Test removal of cancelled orders (InvoiceNo starting with C)."""
        initial_count = len(sample_transaction_data)
        cleaner = DataCleaner(sample_transaction_data)
        cleaner.remove_cancelled_orders()
        
        # Check no cancelled orders remain
        assert not cleaner.df['InvoiceNo'].astype(str).str.startswith('C').any()
        assert len(cleaner.df) < initial_count
    
    def test_remove_duplicates(self, sample_transaction_data):
        """Test removal of duplicate rows."""
        # Add duplicates
        df_with_dups = pd.concat([sample_transaction_data, 
                                   sample_transaction_data.head(10)], ignore_index=True)
        
        cleaner = DataCleaner(df_with_dups)
        initial_count = len(cleaner.df)
        cleaner.remove_duplicates()
        
        assert len(cleaner.df) < initial_count
    
    def test_handle_outliers_iqr(self, clean_sample_data):
        """Test outlier detection using IQR method."""
        # Add outliers
        clean_sample_data.loc[0, 'Quantity'] = 10000
        clean_sample_data.loc[1, 'UnitPrice'] = 10000
        
        cleaner = DataCleaner(clean_sample_data.copy())
        initial_count = len(cleaner.df)
        cleaner.handle_outliers_iqr('Quantity', multiplier=1.5)
        
        # Outliers should be removed
        assert len(cleaner.df) < initial_count
        assert cleaner.df['Quantity'].max() < 10000
    
    def test_convert_dates(self, sample_transaction_data):
        """Test date column conversion to datetime."""
        cleaner = DataCleaner(sample_transaction_data)
        cleaner.convert_dates()
        
        assert cleaner.df['InvoiceDate'].dtype == 'datetime64[ns]'
    
    def test_create_derived_features(self, clean_sample_data):
        """Test creation of derived features."""
        cleaner = DataCleaner(clean_sample_data.copy())
        cleaner.create_derived_features()
        
        # Check derived features exist
        assert 'TotalAmount' in cleaner.df.columns
        assert 'Year' in cleaner.df.columns
        assert 'Month' in cleaner.df.columns
        assert 'Day' in cleaner.df.columns
        assert 'DayOfWeek' in cleaner.df.columns
        assert 'Hour' in cleaner.df.columns
    
    def test_create_derived_features_total_amount(self, clean_sample_data):
        """Test TotalAmount calculation."""
        cleaner = DataCleaner(clean_sample_data.copy())
        cleaner.create_derived_features()
        
        # TotalAmount should equal Quantity * UnitPrice
        expected = clean_sample_data['Quantity'] * clean_sample_data['UnitPrice']
        np.testing.assert_array_almost_equal(
            cleaner.df['TotalAmount'].values, 
            expected.values
        )
    
    def test_get_cleaned_data(self, sample_transaction_data):
        """Test retrieval of cleaned data."""
        cleaner = DataCleaner(sample_transaction_data)
        cleaner.handle_missing_values()
        
        result = cleaner.get_cleaned_data()
        
        assert isinstance(result, pd.DataFrame)
        assert result is cleaner.df
    
    def test_print_cleaning_summary(self, sample_transaction_data, capsys):
        """Test printing of cleaning summary."""
        cleaner = DataCleaner(sample_transaction_data)
        cleaner.handle_missing_values()
        cleaner.print_cleaning_summary()
        
        captured = capsys.readouterr()
        assert "CLEANING SUMMARY" in captured.out
        assert "Original shape:" in captured.out
        assert "Cleaned shape:" in captured.out
        assert "Rows removed:" in captured.out
    
    def test_method_chaining(self, sample_transaction_data):
        """Test that methods can be chained together."""
        cleaner = DataCleaner(sample_transaction_data)
        
        result = (cleaner
                  .convert_dates()
                  .handle_missing_values()
                  .remove_invalid_quantities()
                  .remove_invalid_prices()
                  .remove_duplicates()
                  .create_derived_features())
        
        assert result is cleaner


class TestCleanData:
    """Test cases for clean_data function."""
    
    def test_clean_data_default(self, sample_transaction_data):
        """Test clean_data with default parameters."""
        cleaned = clean_data(sample_transaction_data)
        
        assert isinstance(cleaned, pd.DataFrame)
        assert len(cleaned) > 0
        assert cleaned['CustomerID'].notna().all()
        assert (cleaned['Quantity'] > 0).all()
        assert (cleaned['UnitPrice'] > 0).all()
    
    def test_clean_data_no_outliers(self, sample_transaction_data):
        """Test clean_data with outlier removal disabled."""
        cleaned = clean_data(sample_transaction_data, remove_outliers=False)
        
        # Should still remove invalid values but not outliers
        assert cleaned['CustomerID'].notna().all()
    
    def test_clean_data_adds_derived_features(self, clean_sample_data):
        """Test that clean_data adds derived features."""
        cleaned = clean_data(clean_sample_data)
        
        assert 'TotalAmount' in cleaned.columns
        assert 'Year' in cleaned.columns
        assert 'Month' in cleaned.columns
    
    def test_clean_data_with_duplicate_invoices(self):
        """Test clean_data handles duplicate invoices."""
        data = {
            'InvoiceNo': ['INV001', 'INV001', 'INV002'],
            'CustomerID': ['C001', 'C001', 'C002'],
            'InvoiceDate': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02']),
            'Quantity': [10, 10, 20],
            'UnitPrice': [5.0, 5.0, 10.0],
            'Description': ['Prod1', 'Prod1', 'Prod2'],
        }
        df = pd.DataFrame(data)
        
        cleaned = clean_data(df)
        
        # Duplicates should be removed
        assert len(cleaned) <= 2


class TestGetDataQualityReport:
    """Test cases for get_data_quality_report function."""
    
    def test_get_data_quality_report_basic(self, clean_sample_data):
        """Test basic data quality report generation."""
        report = get_data_quality_report(clean_sample_data)
        
        assert isinstance(report, pd.DataFrame)
        assert 'Column' in report.columns
        assert 'Non-Null Count' in report.columns
        assert 'Null Count' in report.columns
        assert 'Null %' in report.columns
        assert 'Unique Values' in report.columns
        assert 'Data Type' in report.columns
    
    def test_get_data_quality_report_with_nulls(self):
        """Test data quality report with missing values."""
        df = pd.DataFrame({
            'A': [1, 2, None, 4, 5],
            'B': ['x', 'y', 'z', None, 'w'],
            'C': [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        
        report = get_data_quality_report(df)
        
        # Check null counts
        assert report[report['Column'] == 'A']['Null Count'].values[0] == 1
        assert report[report['Column'] == 'B']['Null Count'].values[0] == 1
        assert report[report['Column'] == 'C']['Null Count'].values[0] == 0
    
    def test_get_data_quality_report_null_percentage(self):
        """Test null percentage calculation."""
        df = pd.DataFrame({
            'A': [1, None, None, 4, None],  # 60% null
        })
        
        report = get_data_quality_report(df)
        
        null_pct = report[report['Column'] == 'A']['Null %'].values[0]
        assert null_pct == 60.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
