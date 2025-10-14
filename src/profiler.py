"""Week 2: Core Profiler - Fast data profiling with Polars"""
import polars as pl
from pathlib import Path
from typing import Dict, Any, List
import json


class DataProfiler:
    """Fast profiler for CSV/Parquet files using Polars"""
    
    def __init__(self, file_path: str):
        """
        Initialize profiler with file path
        
        Args:
            file_path: Path to CSV or Parquet file
        """
        self.file_path = Path(file_path)
        self.df: pl.DataFrame = None
        
    def load_data(self, sample_size: int = None) -> None:
        """
        Load data from file
        
        Args:
            sample_size: Optional number of rows to sample (None = all rows)
        """
        if self.file_path.suffix.lower() == '.parquet':
            self.df = pl.read_parquet(self.file_path)
        elif self.file_path.suffix.lower() in ['.csv', '.txt']:
            self.df = pl.read_csv(self.file_path)
        else:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")
        
        if sample_size and len(self.df) > sample_size:
            self.df = self.df.sample(n=sample_size)
    
    def profile_column(self, column: str) -> Dict[str, Any]:
        """
        Profile a single column
        
        Args:
            column: Column name
            
        Returns:
            Dictionary with column statistics
        """
        col_data = self.df[column]
        dtype = str(col_data.dtype)
        
        total_count = len(col_data)
        null_count = col_data.null_count()
        null_percentage = round(null_count / total_count * 100, 2) if total_count > 0 else 0.0
        
        profile = {
            "name": column,
            "dtype": dtype,
            "total_count": total_count,
            "null_count": null_count,
            "null_percentage": null_percentage,
            "unique_count": col_data.n_unique(),
        }
        
        # Numeric columns
        if col_data.dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                               pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                               pl.Float32, pl.Float64]:
            stats = col_data.drop_nulls()
            if len(stats) > 0:
                profile.update({
                    "min": float(stats.min()),
                    "max": float(stats.max()),
                    "mean": float(stats.mean()),
                    "median": float(stats.median()),
                    "std": float(stats.std()) if len(stats) > 1 else 0.0,
                    "negative_count": int((stats < 0).sum()),
                    "zero_count": int((stats == 0).sum()),
                })
        
        # String columns
        elif col_data.dtype == pl.Utf8:
            non_null = col_data.drop_nulls()
            if len(non_null) > 0:
                lengths = non_null.str.len_chars()
                profile.update({
                    "min_length": int(lengths.min()),
                    "max_length": int(lengths.max()),
                    "avg_length": float(lengths.mean()),
                })
                
                # Sample values for small unique counts
                if profile["unique_count"] <= 20:
                    profile["unique_values"] = non_null.unique().to_list()[:20]
        
        # Boolean columns
        elif col_data.dtype == pl.Boolean:
            non_null = col_data.drop_nulls()
            if len(non_null) > 0:
                profile.update({
                    "true_count": int(non_null.sum()),
                    "false_count": int((~non_null).sum()),
                })
        
        # Date/Datetime columns
        elif col_data.dtype in [pl.Date, pl.Datetime]:
            non_null = col_data.drop_nulls()
            if len(non_null) > 0:
                profile.update({
                    "min_date": str(non_null.min()),
                    "max_date": str(non_null.max()),
                })
        
        return profile
    
    def profile_all(self) -> Dict[str, Any]:
        """
        Profile entire dataset
        
        Returns:
            Dictionary with full dataset profile
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        profile = {
            "file_name": self.file_path.name,
            "row_count": len(self.df),
            "column_count": len(self.df.columns),
            "columns": []
        }
        
        for column in self.df.columns:
            col_profile = self.profile_column(column)
            profile["columns"].append(col_profile)
        
        return profile
    
    def to_json(self, output_path: str = None) -> str:
        """
        Export profile to JSON
        
        Args:
            output_path: Optional path to save JSON file
            
        Returns:
            JSON string
        """
        profile = self.profile_all()
        json_str = json.dumps(profile, indent=2)
        
        if output_path:
            Path(output_path).write_text(json_str)
        
        return json_str


def profile_file(file_path: str, sample_size: int = None) -> Dict[str, Any]:
    """
    Convenience function to profile a file
    
    Args:
        file_path: Path to file
        sample_size: Optional sample size
        
    Returns:
        Profile dictionary
    """
    profiler = DataProfiler(file_path)
    profiler.load_data(sample_size=sample_size)
    return profiler.profile_all()

