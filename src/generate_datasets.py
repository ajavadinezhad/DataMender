#!/usr/bin/env python3
"""
Generate test files of various sizes for testing auto-chunking functionality
Creates files from 25K rows to multi-gigabyte sizes
"""

import sys
from pathlib import Path
import polars as pl
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.data_generator import generate_ride_sharing_data


def estimate_file_size_mb(num_rows: int) -> float:
    """Estimate file size in MB for CSV (rough approximation)"""
    return (num_rows * 1024) / (1024 * 1024)


def generate_datasets():
    """Generate test files of various sizes"""
    
    test_sizes = [
        (25000, "25K", "small"),
        (50000, "50K", "small"),
        (100000, "100K", "medium"),
        (250000, "250K", "medium"),
        (500000, "500K", "large"),
        (1000000, "1M", "large"),
        (2500000, "2.5M", "xlarge"),
        (5000000, "5M", "xlarge"),
        (10000000, "10M", "huge"),
        (20000000, "20M", "huge"),
    ]
    
    output_dir = Path("datasets")
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Generating Test Files for Auto-Chunking Testing")
    print("="*60)
    print()
    
    for num_rows, label, category in test_sizes:
        print(f"Generating {label} rows ({category})...", end=" ", flush=True)
        
        try:
            df = generate_ride_sharing_data(num_rows)
            
            # Add some intentional issues for testing (extra layer of manipulation)
            random.seed(42)
            
            df = df.with_columns(
                pl.when(pl.col("fare_amount") % 7 == 0)
                .then(pl.col("fare_amount") * -1)
                .otherwise(pl.col("fare_amount"))
                .alias("fare_amount")
            )
            
            csv_path = output_dir / f"sample_rides_{label.lower()}.csv"
            df.write_csv(csv_path)
            csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
            
            parquet_path = output_dir / f"sample_rides_{label.lower()}.parquet"
            df.write_parquet(parquet_path)
            parquet_size_mb = parquet_path.stat().st_size / (1024 * 1024)
            

            
            print("Done")
            print(f"   CSV: {csv_path.name} ({csv_size_mb:.2f} MB)")
            print(f"   Parquet: {parquet_path.name} ({parquet_size_mb:.2f} MB)")
            print()
            
        except Exception as e:
            print(f"Error: {str(e)}")
            print()
    
    print("="*60)
    print("Test file generation complete!")
    print("="*60)
    print()
    print("Files saved to: datasets/")
    print()
    print("File Size Guide:")
    print("   <500MB: Chunking disabled (auto)")
    print("   500MB-1GB: Chunking enabled, 200K chunks (auto)")
    print("   1GB-2GB: Chunking enabled, 100K chunks (auto)")
    print("   >2GB: Chunking enabled, 50K chunks (auto)")
    print()
    print("Note: Maximum file size is limited to 20M records for safety.")
    print("   For larger files, modify the script or generate them manually.")
    print()


if __name__ == "__main__":
    generate_datasets()
