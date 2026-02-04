"""
Pytest Configuration and Shared Fixtures

Contains fixtures used across all test modules.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_transaction_data():
    """
    Create sample transaction data for testing.
    
    Returns a DataFrame with realistic transaction data including
    common edge cases like missing values, duplicates, etc.
    """
    np.random.seed(42)
    
    n_rows = 500
    dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='H')
    
    data = {
        'InvoiceNo': [f'INV{i:06d}' for i in range(n_rows)],
        'CustomerID': [f'C{np.random.randint(1, 100):04d}' for _ in range(n_rows)],
        'InvoiceDate': dates,
        'Quantity': np.random.randint(1, 50, n_rows),
        'UnitPrice': np.random.uniform(1, 100, n_rows).round(2),
        'Description': [f'Product {np.random.randint(1, 80)}' for _ in range(n_rows)],
        'Country': np.random.choice(['UK', 'USA', 'Germany', 'France', 'Australia'], n_rows),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_rfm_data():
    """
    Create sample RFM data for testing.
    
    Returns a DataFrame with Recency, Frequency, Monetary metrics.
    """
    np.random.seed(42)
    
    n_customers = 100
    data = {
        'CustomerID': [f'C{i:04d}' for i in range(1, n_customers + 1)],
        'Recency': np.random.randint(1, 365, n_customers),
        'Frequency': np.random.randint(1, 50, n_customers),
        'Monetary': np.random.uniform(100, 10000, n_customers).round(2),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_segmented_data(sample_rfm_data):
    """
    Create sample RFM data with segments.
    
    Returns a DataFrame with RFM scores and segments.
    """
    df = sample_rfm_data.copy()
    
    # Assign random scores
    df['R_Score'] = np.random.randint(1, 6, len(df))
    df['F_Score'] = np.random.randint(1, 6, len(df))
    df['M_Score'] = np.random.randint(1, 6, len(df))
    df['RFM_Score'] = df['R_Score'] * 100 + df['F_Score'] * 10 + df['M_Score']
    
    # Assign segments based on score ranges
    def assign_segment(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        score = r + f + m
        
        if score >= 12:
            return 'Champions'
        elif score >= 9:
            return 'Loyal'
        elif score >= 7:
            return 'Potential Loyalist'
        elif score >= 6:
            return 'New Customers'
        elif score >= 5:
            return 'Promising'
        elif score >= 4:
            return 'Need Attention'
        elif score >= 3:
            return 'About to Sleep'
        elif r > 3 and f <= 2:
            return 'At Risk'
        elif r <= 2 and f <= 2:
            return 'Lost'
        else:
            return 'Hibernating'
    
    df['Segment'] = df.apply(assign_segment, axis=1)
    
    return df


@pytest.fixture
def dirty_transaction_data():
    """
    Create transaction data with various data quality issues.
    
    Includes missing values, duplicates, invalid quantities, etc.
    """
    np.random.seed(123)
    
    n_rows = 200
    dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='H')
    
    # Create base data
    data = {
        'InvoiceNo': [f'INV{i:06d}' for i in range(n_rows)],
        'CustomerID': [f'C{np.random.randint(1, 50):04d}' for _ in range(n_rows)],
        'InvoiceDate': dates,
        'Quantity': np.random.randint(-10, 100, n_rows),
        'UnitPrice': np.random.uniform(0, 100, n_rows).round(2),
        'Description': [f'Product {np.random.randint(1, 50)}' for _ in range(n_rows)],
    }
    
    df = pd.DataFrame(data)
    
    # Introduce data quality issues
    # Missing CustomerID (5% of rows)
    df.loc[0:9, 'CustomerID'] = np.nan
    
    # Missing Description (5% of rows)
    df.loc[10:19, 'Description'] = np.nan
    
    # Invalid quantities (negative)
    df.loc[20:29, 'Quantity'] = -999
    
    # Invalid prices (zero or negative)
    df.loc[30:39, 'UnitPrice'] = 0
    df.loc[40:49, 'UnitPrice'] = -5
    
    # Cancelled orders
    df.loc[50:59, 'InvoiceNo'] = f'C{np.random.randint(100000, 200000)}'
    
    # Duplicates
    df = pd.concat([df, df.head(20)], ignore_index=True)
    
    return df


@pytest.fixture
def empty_dataframe():
    """Return an empty DataFrame with expected columns."""
    return pd.DataFrame(columns=[
        'InvoiceNo', 'CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice', 'Description'
    ])


@pytest.fixture
def single_customer_data():
    """Return DataFrame with single customer data."""
    np.random.seed(42)
    
    customer_id = 'C0001'
    n_transactions = 50
    dates = pd.date_range(start='2023-01-01', periods=n_transactions, freq='D')
    
    data = {
        'InvoiceNo': [f'INV{i:06d}' for i in range(n_transactions)],
        'CustomerID': [customer_id] * n_transactions,
        'InvoiceDate': dates,
        'Quantity': np.random.randint(1, 20, n_transactions),
        'UnitPrice': np.random.uniform(5, 50, n_transactions).round(2),
        'Description': [f'Product {np.random.randint(1, 20)}' for _ in range(n_transactions)],
    }
    
    return pd.DataFrame(data)
