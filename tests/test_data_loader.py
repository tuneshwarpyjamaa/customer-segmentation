"""
Unit Tests for Data Loader Module

Tests the memory-efficient data loading functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import DataLoader, load_data_optimized, get_data_info


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    
    n_rows = 1000
    data = {
        'InvoiceNo': [f'INV{i:06d}' for i in range(n_rows)],
        'CustomerID': [f'C{np.random.randint(1, 100):04d}' for _ in range(n_rows)],
        'InvoiceDate': pd.date_range(start='2023-01-01', periods=n_rows, freq='H'),
        'Quantity': np.random.randint(1, 50, n_rows),
        'UnitPrice': np.random.uniform(1, 100, n_rows).round(2),
        'Description': [f'Product {i}' for i in np.random.randint(1, 100, n_rows)],
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_file(sample_dataframe):
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_dataframe.to_csv(f.name, index=False)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_excel_file(sample_dataframe):
    """Create a temporary Excel file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        sample_dataframe.to_excel(f.name, index=False)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_init_default_chunk_size(self):
        """Test DataLoader initialization with default chunk size."""
        loader = DataLoader()
        assert loader.chunk_size == 50000
    
    def test_init_custom_chunk_size(self):
        """Test DataLoader initialization with custom chunk size."""
        loader = DataLoader(chunk_size=25000)
        assert loader.chunk_size == 25000
    
    def test_optimize_dtypes_integers(self):
        """Test dtype optimization for integer columns."""
        df = pd.DataFrame({
            'col_int64': np.array([1, 2, 3, 4, 5], dtype='int64'),
            'col_int32': np.array([1, 2, 3, 4, 5], dtype='int32'),
        })
        
        loader = DataLoader()
        optimized = loader.optimize_dtypes(df.copy())
        
        # Check that memory-optimized types are used
        assert optimized['col_int64'].dtype in ['uint8', 'uint16', 'uint32']
    
    def test_optimize_dtypes_floats(self):
        """Test dtype optimization for float columns."""
        df = pd.DataFrame({
            'col_float64': np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype='float64'),
        })
        
        loader = DataLoader()
        optimized = loader.optimize_dtypes(df.copy())
        
        # Check that float32 is used
        assert optimized['col_float64'].dtype == 'float32'
    
    def test_optimize_dtypes_categorical(self):
        """Test dtype optimization for low-cardinality string columns."""
        df = pd.DataFrame({
            'low_cardinality': ['A', 'B', 'A', 'B', 'A'] * 20,  # 33% unique
            'high_cardinality': [f'Item_{i}' for i in range(100)],  # 100% unique
        })
        
        loader = DataLoader()
        optimized = loader.optimize_dtypes(df.copy())
        
        # Low cardinality should be categorical
        assert optimized['low_cardinality'].dtype == 'category'
        # High cardinality should remain object
        assert optimized['high_cardinality'].dtype == 'object'
    
    def test_load_csv_chunked(self, temp_csv_file):
        """Test loading CSV file in chunks."""
        loader = DataLoader(chunk_size=100)
        df = loader.load_csv_chunked(temp_csv_file)
        
        assert len(df) == 1000
        assert 'InvoiceNo' in df.columns
        assert 'CustomerID' in df.columns
    
    def test_load_csv_with_usecols(self, temp_csv_file):
        """Test loading CSV with specific columns."""
        loader = DataLoader()
        df = loader.load_csv_chunked(temp_csv_file, usecols=['InvoiceNo', 'CustomerID'])
        
        assert list(df.columns) == ['InvoiceNo', 'CustomerID']
        assert len(df) == 1000
    
    def test_load_excel_optimized(self, temp_excel_file):
        """Test loading Excel file with optimization."""
        loader = DataLoader()
        df = loader.load_excel_optimized(temp_excel_file)
        
        assert len(df) == 1000
        assert 'InvoiceNo' in df.columns
        assert 'CustomerID' in df.columns
    
    def test_load_excel_with_sheet_name(self, temp_excel_file):
        """Test loading specific sheet from Excel file."""
        loader = DataLoader()
        df = loader.load_excel_optimized(temp_excel_file, sheet_name=0)
        
        assert len(df) == 1000


class TestLoadDataOptimized:
    """Test cases for load_data_optimized function."""
    
    def test_load_csv_format(self, temp_csv_file):
        """Test loading data with CSV format."""
        df = load_data_optimized(temp_csv_file, file_format='csv')
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1000
    
    def test_load_excel_format(self, temp_excel_file):
        """Test loading data with Excel format."""
        df = load_data_optimized(temp_excel_file, file_format='excel')
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1000
    
    def test_invalid_file_format(self):
        """Test error handling for invalid file format."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_data_optimized("test.parquet", file_format='parquet')
    
    def test_custom_chunk_size(self, temp_csv_file):
        """Test loading with custom chunk size."""
        df = load_data_optimized(temp_csv_file, file_format='csv', chunk_size=100)
        assert len(df) == 1000


class TestGetDataInfo:
    """Test cases for get_data_info function."""
    
    def test_get_data_info_basic(self, sample_dataframe, capsys):
        """Test basic data info output."""
        get_data_info(sample_dataframe)
        
        captured = capsys.readouterr()
        assert "DATA INFORMATION" in captured.out
        assert "Shape:" in captured.out
        assert "Memory Usage:" in captured.out
        assert "Column Information:" in captured.out
        assert "Missing Values:" in captured.out
    
    def test_get_data_info_with_nulls(self):
        """Test data info with missing values."""
        df = pd.DataFrame({
            'A': [1, 2, None, 4, 5],
            'B': ['x', 'y', 'z', None, 'w'],
        })
        
        get_data_info(df)
    
    def test_get_data_info_duplicates(self):
        """Test data info with duplicate rows."""
        df = pd.DataFrame({
            'A': [1, 2, 2, 4, 5],
            'B': ['x', 'y', 'y', 'z', 'w'],
        })
        
        get_data_info(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
