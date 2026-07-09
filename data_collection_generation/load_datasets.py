"""
Dataset Loader for LLM Red-Teaming Prompts Capstone Project
Reads all CSV files and displays comprehensive statistics and data
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')


class DatasetLoader:
    def __init__(self, base_path: str = "."):
        """Initialize the dataset loader"""
        self.base_path = Path(base_path)
        self.datasets = {}
        self.dataset_info = {}
        
    def find_csv_files(self) -> List[Path]:
        """Find all CSV files in the repository"""
        csv_files = list(self.base_path.glob("**/*.csv"))
        return sorted(csv_files)
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files into a dictionary of DataFrames"""
        csv_files = self.find_csv_files()
        
        print(f"Found {len(csv_files)} CSV file(s)\n")
        print("=" * 80)
        
        for csv_file in csv_files:
            try:
                relative_path = csv_file.relative_to(self.base_path)
                print(f"\nLoading: {relative_path}")
                
                # Try different encodings
                try:
                    df = pd.read_csv(csv_file)
                except UnicodeDecodeError:
                    df = pd.read_csv(csv_file, encoding='latin-1')
                
                dataset_name = csv_file.stem
                self.datasets[dataset_name] = df
                
                # Store metadata
                self.dataset_info[dataset_name] = {
                    'path': str(relative_path),
                    'rows': len(df),
                    'columns': list(df.columns),
                    'shape': df.shape
                }
                
                print(f"  ✓ Loaded successfully")
                print(f"    Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                print(f"    Columns: {', '.join(df.columns.tolist())}")
                
            except Exception as e:
                print(f"  ✗ Error loading {csv_file}: {str(e)}")
        
        print("\n" + "=" * 80)
        return self.datasets
    
    def display_summary(self):
        """Display summary statistics for all datasets"""
        print("\n" + "=" * 80)
        print("DATASET SUMMARY")
        print("=" * 80)
        
        total_rows = sum(info['rows'] for info in self.dataset_info.values())
        
        print(f"\nTotal Datasets Loaded: {len(self.datasets)}")
        print(f"Total Rows Across All Datasets: {total_rows}")
        print(f"\nDataset Details:")
        print("-" * 80)
        
        for dataset_name, info in self.dataset_info.items():
            print(f"\n{dataset_name}")
            print(f"  Path: {info['path']}")
            print(f"  Dimensions: {info['rows']} rows × {info['columns'].__len__()} columns")
            print(f"  Columns: {', '.join(info['columns'])}")
    
    def display_dataset(self, dataset_name: str, head: int = 5, show_info: bool = True):
        """Display a specific dataset"""
        if dataset_name not in self.datasets:
            print(f"Dataset '{dataset_name}' not found.")
            print(f"Available datasets: {list(self.datasets.keys())}")
            return
        
        df = self.datasets[dataset_name]
        
        print("\n" + "=" * 80)
        print(f"DATASET: {dataset_name}")
        print("=" * 80)
        
        if show_info:
            print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"\nColumn Information:")
            print(df.info())
            print(f"\nData Types:\n{df.dtypes}")
            print(f"\nMissing Values:\n{df.isnull().sum()}")
        
        print(f"\n\nFirst {head} rows:")
        print("-" * 80)
        print(df.head(head).to_string())
        
        if len(df) > head:
            print(f"\n\nLast {head} rows:")
            print("-" * 80)
            print(df.tail(head).to_string())
    
    def display_all_datasets(self, head: int = 3):
        """Display preview of all datasets"""
        for dataset_name in self.datasets.keys():
            self.display_dataset(dataset_name, head=head, show_info=False)
            print("\n")
    
    def get_dataset_statistics(self, dataset_name: str):
        """Get detailed statistics for a dataset"""
        if dataset_name not in self.datasets:
            print(f"Dataset '{dataset_name}' not found.")
            return None
        
        df = self.datasets[dataset_name]
        
        print("\n" + "=" * 80)
        print(f"STATISTICS: {dataset_name}")
        print("=" * 80)
        
        print(f"\nBasic Statistics:")
        print(df.describe(include='all').to_string())
        
        # Column-specific statistics
        for col in df.columns:
            print(f"\n\n{col}:")
            print(f"  Data Type: {df[col].dtype}")
            print(f"  Non-null Count: {df[col].count()}")
            print(f"  Null Count: {df[col].isnull().sum()}")
            
            if df[col].dtype == 'object':
                print(f"  Unique Values: {df[col].nunique()}")
                print(f"  Most Common Value: {df[col].value_counts().index[0]} ({df[col].value_counts().values[0]} occurrences)")
    
    def export_summary_report(self, output_file: str = "dataset_report.txt"):
        """Export a comprehensive report of all datasets"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LLM RED-TEAMING PROMPTS DATASET REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total Datasets: {len(self.datasets)}\n")
            f.write(f"Total Rows: {sum(info['rows'] for info in self.dataset_info.values())}\n\n")
            
            for dataset_name, info in self.dataset_info.items():
                f.write(f"\n{'=' * 80}\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"{'=' * 80}\n")
                f.write(f"Path: {info['path']}\n")
                f.write(f"Shape: {info['rows']} rows × {len(info['columns'])} columns\n")
                f.write(f"Columns: {', '.join(info['columns'])}\n\n")
                
                df = self.datasets[dataset_name]
                f.write(f"Preview (first 10 rows):\n")
                f.write(df.head(10).to_string())
                f.write("\n\n")
        
        print(f"\nReport exported to: {output_file}")
    
    def list_datasets(self):
        """List all loaded datasets"""
        print("\n" + "=" * 80)
        print("LOADED DATASETS")
        print("=" * 80)
        
        if not self.datasets:
            print("No datasets loaded.")
            return
        
        for i, (dataset_name, df) in enumerate(self.datasets.items(), 1):
            print(f"\n{i}. {dataset_name}")
            print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"   Columns: {', '.join(df.columns.tolist())}")


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("LLM RED-TEAMING PROMPTS DATASET LOADER")
    print("=" * 80)
    
    # Initialize loader
    loader = DatasetLoader()
    
    # Load all datasets
    datasets = loader.load_all_datasets()
    
    # Display summary
    loader.display_summary()
    
    # List all datasets
    loader.list_datasets()
    
    # Display detailed information for each dataset
    print("\n\nDisplaying detailed information for each dataset...\n")
    loader.display_all_datasets(head=5)
    
    # Export report
    loader.export_summary_report("dataset_report.txt")
    
    print("\n" + "=" * 80)
    print("DATASET LOADING COMPLETE")
    print("=" * 80)
    
    return loader


if __name__ == "__main__":
    loader = main()
    
    # Interactive mode
    print("\n\nInteractive Mode - You can now inspect datasets:")
    print("Usage examples:")
    print("  loader.display_dataset('baseline_prompts')")
    print("  loader.get_dataset_statistics('baseline_prompts')")
    print("  loader.datasets['baseline_prompts'].head(10)")
    print("\nAvailable datasets:")
    for name in loader.datasets.keys():
        print(f"  - {name}")
