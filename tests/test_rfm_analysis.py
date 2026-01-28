"""
Unit Tests for RFM Analysis Module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rfm_analysis import (RFMAnalyzer, calculate_rfm, segment_customers,
                          get_segment_summary, identify_high_value_customers)


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing."""
    np.random.seed(42)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    customers = [f'C{i:04d}' for i in range(1, 101)]
    
    data = []
    for customer in customers:
        n_transactions = np.random.randint(1, 20)
        for _ in range(n_transactions):
            date = np.random.choice(dates)
            quantity = np.random.randint(1, 50)
            price = np.random.uniform(1, 100)
            data.append({
                'CustomerID': customer,
                'InvoiceNo': f'INV{len(data):06d}',
                'InvoiceDate': date,
                'Quantity': quantity,
                'UnitPrice': price,
                'TotalAmount': quantity * price
            })
    
    df = pd.DataFrame(data)
    return df


def test_rfm_analyzer_initialization(sample_transaction_data):
    """Test RFMAnalyzer initialization."""
    analyzer = RFMAnalyzer(sample_transaction_data)
    
    assert analyzer.df is not None
    assert analyzer.customer_col == 'CustomerID'
    assert analyzer.date_col == 'InvoiceDate'
    assert analyzer.amount_col == 'TotalAmount'
    assert analyzer.reference_date is not None


def test_calculate_rfm_metrics(sample_transaction_data):
    """Test RFM metrics calculation."""
    analyzer = RFMAnalyzer(sample_transaction_data)
    rfm = analyzer.calculate_rfm_metrics()
    
    # Check structure
    assert 'CustomerID' in rfm.columns
    assert 'Recency' in rfm.columns
    assert 'Frequency' in rfm.columns
    assert 'Monetary' in rfm.columns
    
    # Check data types
    assert rfm['Recency'].dtype in ['uint16', 'int64']
    assert rfm['Frequency'].dtype in ['uint16', 'int64']
    assert rfm['Monetary'].dtype in ['float32', 'float64']
    
    # Check values
    assert len(rfm) > 0
    assert (rfm['Recency'] >= 0).all()
    assert (rfm['Frequency'] > 0).all()
    assert (rfm['Monetary'] > 0).all()


def test_assign_rfm_scores(sample_transaction_data):
    """Test RFM score assignment."""
    analyzer = RFMAnalyzer(sample_transaction_data)
    rfm = analyzer.calculate_rfm_metrics()
    rfm_scored = analyzer.assign_rfm_scores(rfm, n_quantiles=5)
    
    # Check score columns exist
    assert 'R_Score' in rfm_scored.columns
    assert 'F_Score' in rfm_scored.columns
    assert 'M_Score' in rfm_scored.columns
    assert 'RFM_Score' in rfm_scored.columns
    assert 'RFM_Value' in rfm_scored.columns
    
    # Check score ranges
    assert rfm_scored['R_Score'].between(1, 5).all()
    assert rfm_scored['F_Score'].between(1, 5).all()
    assert rfm_scored['M_Score'].between(1, 5).all()


def test_create_segments(sample_transaction_data):
    """Test customer segmentation."""
    analyzer = RFMAnalyzer(sample_transaction_data)
    rfm = analyzer.calculate_rfm_metrics()
    rfm_scored = analyzer.assign_rfm_scores(rfm, n_quantiles=5)
    rfm_segmented = analyzer.create_segments(rfm_scored)
    
    # Check segment column exists
    assert 'Segment' in rfm_segmented.columns
    
    # Check all customers are segmented
    assert rfm_segmented['Segment'].notna().all()
    
    # Check segment types
    valid_segments = ['Champions', 'Loyal', 'Potential Loyalist', 'New Customers',
                     'Promising', 'Need Attention', 'About to Sleep', 'At Risk',
                     'Cannot Lose', 'Hibernating', 'Lost']
    assert rfm_segmented['Segment'].isin(valid_segments).all()


def test_calculate_rfm_function(sample_transaction_data):
    """Test the calculate_rfm main function."""
    rfm = calculate_rfm(sample_transaction_data)
    
    assert isinstance(rfm, pd.DataFrame)
    assert len(rfm) > 0
    assert 'Recency' in rfm.columns
    assert 'Frequency' in rfm.columns
    assert 'Monetary' in rfm.columns


def test_segment_customers_function(sample_transaction_data):
    """Test the segment_customers main function."""
    rfm = calculate_rfm(sample_transaction_data)
    rfm_segmented = segment_customers(rfm)
    
    assert isinstance(rfm_segmented, pd.DataFrame)
    assert 'Segment' in rfm_segmented.columns
    assert 'R_Score' in rfm_segmented.columns
    assert 'F_Score' in rfm_segmented.columns
    assert 'M_Score' in rfm_segmented.columns


def test_get_segment_summary(sample_transaction_data):
    """Test segment summary generation."""
    rfm = calculate_rfm(sample_transaction_data)
    rfm_segmented = segment_customers(rfm)
    summary = get_segment_summary(rfm_segmented)
    
    assert isinstance(summary, pd.DataFrame)
    assert 'Segment' in summary.columns
    assert 'Customer_Count' in summary.columns
    assert 'Avg_Recency' in summary.columns
    assert 'Avg_Frequency' in summary.columns
    assert 'Avg_Monetary' in summary.columns
    assert 'Total_Revenue' in summary.columns


def test_identify_high_value_customers(sample_transaction_data):
    """Test high-value customer identification."""
    rfm = calculate_rfm(sample_transaction_data)
    rfm_segmented = segment_customers(rfm)
    top_customers = identify_high_value_customers(rfm_segmented, top_n=10)
    
    assert isinstance(top_customers, pd.DataFrame)
    assert len(top_customers) == 10
    assert 'CustomerID' in top_customers.columns
    assert 'Monetary' in top_customers.columns
    
    # Check that they're sorted by Monetary value
    assert (top_customers['Monetary'].values[:-1] >= 
            top_customers['Monetary'].values[1:]).all()


def test_rfm_with_empty_dataframe():
    """Test RFM calculation with edge cases."""
    # Empty DataFrame
    empty_df = pd.DataFrame(columns=['CustomerID', 'InvoiceDate', 'InvoiceNo', 'TotalAmount'])
    
    with pytest.raises(Exception):
        calculate_rfm(empty_df)


def test_rfm_score_ranges(sample_transaction_data):
    """Test that RFM scores are within valid ranges."""
    rfm = calculate_rfm(sample_transaction_data)
    rfm_segmented = segment_customers(rfm, n_quantiles=5)
    
    # All scores should be between 1 and 5
    assert rfm_segmented['R_Score'].min() >= 1
    assert rfm_segmented['R_Score'].max() <= 5
    assert rfm_segmented['F_Score'].min() >= 1
    assert rfm_segmented['F_Score'].max() <= 5
    assert rfm_segmented['M_Score'].min() >= 1
    assert rfm_segmented['M_Score'].max() <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
