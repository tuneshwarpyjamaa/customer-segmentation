"""
Memory-Efficient Data Loader

Optimized data loading with chunked processing and reduced memory footprint.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import gc
from typing import Optional, Dict, Any
from tqdm import tqdm


class DataLoader:
    """
    Efficient data loader with memory optimization techniques.
    """
    
    def __init__(self, chunk_size: int = 50000):
        """
        Initialize DataLoader.
        
        Args:
            chunk_size: Number of rows to process at once
        """
        self.chunk_size = chunk_size
        
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize data types to reduce memory usage.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with optimized dtypes
        """
        # Optimize integer columns
        int_cols = df.select_dtypes(include=['int64', 'int32']).columns
        for col in int_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:  # Unsigned integers
                if col_max < 255:
                    df[col] = df[col].astype('uint8')
                elif col_max < 65535:
                    df[col] = df[col].astype('uint16')
                elif col_max < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:  # Signed integers
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype('int8')
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype('int16')
                elif col_min > -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype('int32')
        
        # Optimize float columns
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = df[col].astype('float32')
        
        # Convert to categorical for low-cardinality string columns
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            if col != 'InvoiceDate':  # Skip datetime columns
                num_unique = df[col].nunique()
                num_total = len(df[col])
                if num_unique / num_total < 0.5:  # Less than 50% unique
                    df[col] = df[col].astype('category')
        
        return df
    
    def load_excel_optimized(
        self, 
        filepath: str,
        sheet_name: str = 0,
        usecols: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Load Excel file with memory optimization.
        
        Args:
            filepath: Path to Excel file
            sheet_name: Sheet name or index
            usecols: Columns to load
            
        Returns:
            Optimized DataFrame
        """
        print(f"Loading data from {filepath}...")
        
        # Read in chunks if file is large
        try:
            # Try to read the entire file at once
            df = pd.read_excel(
                filepath,
                sheet_name=sheet_name,
                usecols=usecols,
                engine='openpyxl'
            )
            
            print(f"Loaded {len(df):,} rows")
            print(f"Memory usage before optimization: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Optimize data types
            df = self.optimize_dtypes(df)
            
            print(f"Memory usage after optimization: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Force garbage collection
            gc.collect()
            
            return df
            
        except Exception as e:
            print(f"Error loading file: {e}")
            raise
    
    def load_csv_chunked(
        self,
        filepath: str,
        usecols: Optional[list] = None,
        dtype: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Load CSV file in chunks for memory efficiency.
        
        Args:
            filepath: Path to CSV file
            usecols: Columns to load
            dtype: Data type specifications
            
        Returns:
            Optimized DataFrame
        """
        print(f"Loading CSV data from {filepath}...")
        
        chunks = []
        total_rows = 0
        
        # Read in chunks
        chunk_iter = pd.read_csv(
            filepath,
            usecols=usecols,
            dtype=dtype,
            chunksize=self.chunk_size,
            low_memory=False
        )
        
        for chunk in tqdm(chunk_iter, desc="Processing chunks"):
            chunk = self.optimize_dtypes(chunk)
            chunks.append(chunk)
            total_rows += len(chunk)
        
        # Concatenate all chunks
        df = pd.concat(chunks, ignore_index=True)
        
        print(f"Loaded {total_rows:,} rows")
        print(f"Final memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Force garbage collection
        del chunks
        gc.collect()
        
        return df


def load_data_optimized(
    filepath: str,
    file_format: str = 'excel',
    chunk_size: int = 50000
) -> pd.DataFrame:
    """
    Main function to load data with optimization.
    
    Args:
        filepath: Path to data file
        file_format: 'excel' or 'csv'
        chunk_size: Chunk size for processing
        
    Returns:
        Optimized DataFrame
    """
    loader = DataLoader(chunk_size=chunk_size)
    
    if file_format.lower() == 'excel':
        return loader.load_excel_optimized(filepath)
    elif file_format.lower() == 'csv':
        return loader.load_csv_chunked(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def get_data_info(df: pd.DataFrame) -> None:
    """
    Print detailed information about the DataFrame.
    
    Args:
        df: Input DataFrame
    """
    print("\n" + "="*80)
    print("DATA INFORMATION")
    print("="*80)
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumn Information:")
    print(df.dtypes)
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nDuplicate Rows: {df.duplicated().sum():,}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Test the loader
    data_path = Path(__file__).parent.parent / "data" / "raw" / "OnlineRetail.xlsx"
    
    if data_path.exists():
        df = load_data_optimized(str(data_path))
        get_data_info(df)
    else:
        print(f"Data file not found at {data_path}")
        print("Please download the dataset first using scripts/download_data.py")
