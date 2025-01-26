from datasets import load_dataset
import pandas as pd
import re
from pathlib import Path
import logging
from typing import Optional, Dict, Union
from tqdm import tqdm
import os
import argparse
import sys

class DatasetPreprocessor:
    """
    Preprocessor for loading and processing Python code dataset from Hugging Face.
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize the preprocessor.
        
        Args:
            output_dir: Directory to save processed datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def extract_python_code(self, text: str) -> Optional[str]:
        """Extract Python code from text enclosed in ```python``` tags."""
        try:
            if match := re.search(r'```python(.*?)```', text, re.DOTALL):
                code = match.group(1).strip()
                if any(keyword in code for keyword in ['def ', 'class ', 'import ', 'print', 'for', 'while']):
                    return code
                self.logger.warning("Extracted content might not be valid Python code")
            return None
        except Exception as e:
            self.logger.error(f"Error extracting Python code: {str(e)}")
            return None
            
    def process_dataset(self, args: argparse.Namespace) -> pd.DataFrame:
        """
        Load and process dataset from Hugging Face.
        
        Args:
            args: Command line arguments containing dataset_name and batch_size
        """
        self.logger.info(f"Loading dataset: {args.dataset}")
        dataset = load_dataset(args.dataset)
        
        df = pd.DataFrame(dataset['train'])
        initial_size = len(df)
        self.logger.info(f"Initial dataset size: {initial_size}")
        
        df = df.drop(['instruction', 'system'], axis=1)
        
        processed_codes = []
        for i in tqdm(range(0, len(df), args.batch_size), desc="Processing dataset"):
            batch = df.iloc[i:i + args.batch_size]
            batch_codes = batch['output'].apply(self.extract_python_code)
            processed_codes.extend(batch_codes)
        
        result_df = pd.DataFrame({'English_code': processed_codes})
        result_df = result_df.dropna()
        result_df = result_df.drop_duplicates()
        
        final_size = len(result_df)
        self.logger.info(f"Final dataset size: {final_size}")
        self.logger.info(f"Removed {initial_size - final_size} invalid entries")
        
        return result_df
            
    def save_dataset(self, df: pd.DataFrame, filename: str = "python_code_dataset.csv") -> Path:
        """Save processed dataset to CSV file."""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        self.logger.info(f"Dataset saved to: {output_path}")
        return output_path
        
    def get_dataset_stats(self, df: pd.DataFrame) -> Dict:
        """Generate statistics about the processed dataset."""
        return {
            'total_samples': len(df),
            'unique_samples': df['English_code'].nunique(),
            'average_code_length': df['English_code'].str.len().mean(),
            'min_code_length': df['English_code'].str.len().min(),
            'max_code_length': df['English_code'].str.len().max(),
            'null_values': df['English_code'].isnull().sum()
        }
        
    def process_and_save_dataset(self, args: argparse.Namespace) -> Dict:
        """
        Complete pipeline to process dataset and save results.
        
        Args:
            args: Command line arguments
        """
        try:
            # Check if dataset exists and force_preprocess is not set
            output_path = self.output_dir / "python_code_dataset.csv"
            if output_path.exists() and not args.force_preprocess:
                self.logger.info("Dataset already exists. Use --force-preprocess to regenerate.")
                
                # Load existing statistics if available
                stats_file = self.output_dir / "dataset_stats.json"
                if stats_file.exists():
                    stats = pd.read_json(stats_file, typ='series').to_dict()
                else:
                    # Generate stats from existing file
                    df = pd.read_csv(output_path)
                    stats = self.get_dataset_stats(df)
                
                return {
                    'success': True,
                    'output_path': str(output_path),
                    'stats': stats,
                    'skipped': True
                }
            
            # Process dataset
            df = self.process_dataset(args)
            
            # Save to file
            output_path = self.save_dataset(df)
            
            # Generate and save statistics
            stats = self.get_dataset_stats(df)
            stats_file = self.output_dir / "dataset_stats.json"
            pd.Series(stats).to_json(stats_file)
            
            return {
                'success': True,
                'output_path': str(output_path),
                'stats': stats,
                'skipped': False
            }
            
        except Exception as e:
            self.logger.error(f"Error processing dataset: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

# def run_preprocessing(args: argparse.Namespace) -> bool:
#     """
#     Run preprocessing pipeline from command line arguments.
    
#     Args:
#         args: Command line arguments
        
#     Returns:
#         bool: True if preprocessing was successful
#     """
#     # Configure logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )
    
#     # Initialize preprocessor
#     preprocessor = DatasetPreprocessor('data/input/samples')
    
#     # Process dataset
#     results = preprocessor.process_and_save_dataset(args)
    
#     if results['success']:
#         if results.get('skipped'):
#             print("\nUsing existing dataset.")
#         else:
#             print("\nDataset processing completed successfully!")
#         print(f"Dataset location: {results['output_path']}")
#         print("\nDataset Statistics:")
#         for key, value in results['stats'].items():
#             print(f"{key}: {value}")
#         return True
#     else:
#         print(f"\nError processing dataset: {results['error']}")
#         return False

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Process Python code dataset')
#     parser.add_argument('--dataset', type=str, default="jtatman/python-code-dataset-500k",
#                       help='HuggingFace dataset name')
#     parser.add_argument('--batch-size', type=int, default=1000,
#                       help='Batch size for processing')
#     parser.add_argument('--force-preprocess', action='store_true',
#                       help='Force preprocessing even if dataset exists')
    
#     args = parser.parse_args()
#     success = run_preprocessing(args)
#     sys.exit(0 if success else 1)