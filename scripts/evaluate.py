#!/usr/bin/env python3

import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import ast
import tokenize
from io import StringIO
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import re
from pathlib import Path
import json
from typing import List, Dict, Optional, Union
import logging
import traceback

logger = logging.getLogger(__name__)

class CodeEvaluator:
    """Evaluates translated code using multiple metrics."""
    
    def __init__(self):
        """Initialize code evaluator with required models."""
        try:
            self.semantic_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            self.structure_tokens = {'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except'}
        except Exception as e:
            logger.error(f"Failed to initialize evaluator: {str(e)}")
            raise

    def evaluate_code_pair(self, original: str, translated: str) -> Dict[str, float]:
        """
        Evaluate a pair of original and translated code.
        
        Args:
            original: Original source code
            translated: Translated/back-translated code
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            metrics = {
                'bleu_score': self.calculate_bleu(original, translated),
                'syntax_valid': self.check_syntax(translated),
                'structure_score': self.compare_structure(original, translated),
                'semantic_score': self.calculate_semantic_similarity(original, translated),
                'token_match': self.compare_tokens(original, translated)
            }
            
            # Calculate overall score
            valid_scores = [
                score for score in [
                    metrics['bleu_score'],
                    float(metrics['syntax_valid']),
                    metrics['structure_score'],
                    metrics['semantic_score'],
                    metrics['token_match']
                ] if score is not None
            ]
            
            metrics['overall_score'] = np.mean(valid_scores) if valid_scores else 0.0
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return self._create_error_metrics()

    def calculate_bleu(self, original: str, translated: str) -> float:
        """Calculate BLEU score between original and translated code."""
        try:
            def tokenize_code(code: str) -> List[str]:
                tokens = []
                try:
                    for tok in tokenize.generate_tokens(StringIO(code).readline):
                        if tok.type in {tokenize.NAME, tokenize.STRING, tokenize.NUMBER, tokenize.OP}:
                            tokens.append(tok.string)
                except:
                    # Fallback to simple tokenization
                    tokens = code.split()
                return tokens

            reference = [tokenize_code(original)]
            candidate = tokenize_code(translated)
            
            return sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
        except Exception as e:
            logger.error(f"BLEU score calculation failed: {str(e)}")
            return 0.0

    def check_syntax(self, code: str) -> bool:
        """Check if code has valid Python syntax."""
        try:
            ast.parse(code)
            return True
        except:
            return False

    def compare_structure(self, original: str, translated: str) -> float:
        """Compare structural similarity of code."""
        try:
            def extract_structure(code: str) -> Dict[str, List[str]]:
                structure = {'functions': [], 'classes': [], 'control': []}
                try:
                    tree = ast.parse(code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            structure['functions'].append(node.name)
                        elif isinstance(node, ast.ClassDef):
                            structure['classes'].append(node.name)
                        elif isinstance(node, (ast.If, ast.For, ast.While)):
                            structure['control'].append(type(node).__name__)
                except:
                    pass
                return structure

            orig_struct = extract_structure(original)
            trans_struct = extract_structure(translated)
            
            scores = []
            for key in orig_struct:
                orig_items = set(orig_struct[key])
                trans_items = set(trans_struct[key])
                if orig_items or trans_items:
                    similarity = len(orig_items & trans_items) / max(len(orig_items | trans_items), 1)
                    scores.append(similarity)
                    
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Structure comparison failed: {str(e)}")
            return 0.0

    def calculate_semantic_similarity(self, original: str, translated: str) -> float:
        """Calculate semantic similarity using embeddings."""
        try:
            # Get embeddings
            orig_embed = self.semantic_model.encode([original])[0]
            trans_embed = self.semantic_model.encode([translated])[0]
            
            # Calculate cosine similarity
            similarity = 1 - cosine(orig_embed, trans_embed)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {str(e)}")
            return 0.0

    def compare_tokens(self, original: str, translated: str) -> float:
        """Compare token distributions."""
        try:
            def get_token_counts(code: str) -> Dict[str, int]:
                counts = {'total': 0, 'names': 0, 'strings': 0, 'numbers': 0, 'operators': 0}
                try:
                    for tok in tokenize.generate_tokens(StringIO(code).readline):
                        counts['total'] += 1
                        if tok.type == tokenize.NAME:
                            counts['names'] += 1
                        elif tok.type == tokenize.STRING:
                            counts['strings'] += 1
                        elif tok.type == tokenize.NUMBER:
                            counts['numbers'] += 1
                        elif tok.type == tokenize.OP:
                            counts['operators'] += 1
                except:
                    pass
                return counts

            orig_counts = get_token_counts(original)
            trans_counts = get_token_counts(translated)
            
            if orig_counts['total'] == 0 and trans_counts['total'] == 0:
                return 1.0
            elif orig_counts['total'] == 0 or trans_counts['total'] == 0:
                return 0.0
                
            ratios = []
            for key in ['names', 'strings', 'numbers', 'operators']:
                orig = orig_counts[key]
                trans = trans_counts[key]
                if orig or trans:
                    ratio = min(orig, trans) / max(orig, trans)
                    ratios.append(ratio)
                    
            return np.mean(ratios) if ratios else 0.0
            
        except Exception as e:
            logger.error(f"Token comparison failed: {str(e)}")
            return 0.0

    def _create_error_metrics(self) -> Dict[str, float]:
        """Create error metrics dictionary."""
        return {
            'bleu_score': 0.0,
            'syntax_valid': False,
            'structure_score': 0.0,
            'semantic_score': 0.0,
            'token_match': 0.0,
            'overall_score': 0.0
        }

class TranslationEvaluator:
    """Handles evaluation of translation files."""
    
    def __init__(self, args, isStage1: bool = False):
        """Initialize with command line arguments."""
        self.args = args
        self.evaluator = CodeEvaluator()
        self.eval_dir = Path("data/output/evaluation")
        self.isStage1 = isStage1  
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_translations(self, chunk_size: int = 0) -> bool:
        """
        Evaluate translations for a specific chunk size.
        """
        try:
            if not self.isStage1:
                logger.info(f"\nEvaluating translations for chunk size {chunk_size}")
                
                # Get input file
                back_trans_file = Path("data/output/back_translation/stage2/final_back_translation") / \
                    f"Final_back_translation_{self.args.stage2}_{self.args.source_lang}_{self.args.target_lang}_{self.args.stage2_samples}_{chunk_size}.csv"
                    
                if not back_trans_file.exists():
                    logger.error(f"Back translation file not found: {back_trans_file}")
                    return False
                    
                # Read translations
                try:
                    df = pd.read_csv(back_trans_file)
                    logger.info(f"Found {len(df)} translations to evaluate")
                    logger.info(f"Columns in file: {df.columns.tolist()}")
                except Exception as e:
                    logger.error(f"Error reading file {back_trans_file}: {str(e)}")
                    return False
            else:
                logger.info(f"\nEvaluating translations for stage 1")

                # Get input file
                back_trans_file = Path("data/output/back_translation/stage1/final_back_translation") / \
                    f"final_back_translation_{self.args.stage1}_{self.args.source_lang}_{self.args.target_lang}_{self.args.start_index}_{self.args.stage1_samples}_{self.args.stage2_samples}.csv"
                
                if not back_trans_file.exists():
                    logger.error(f"Back translation file not found: {back_trans_file}")
                    return False
                
                # Read translations
                try:
                    df = pd.read_csv(back_trans_file)
                    logger.info(f"Found {len(df)} translations to evaluate")
                    logger.info(f"Columns in file: {df.columns.tolist()}")
                except Exception as e:
                    logger.error(f"Error reading file {back_trans_file}: {str(e)}")
                    return False
                
            
            results = []
            for idx, row in df.iterrows():
                try:
                    # Extract code using proper column names
                    if not self.isStage1:
                        original = row['Original Code']
                        translated = row[f'{self.args.stage2}_partial_translated_code']
                        back_translated = row[f'{self.args.stage2}_back_translated_code']
                    else:
                        original = row['English_code']
                        translated = row['Partial_translated_code']
                        back_translated = row[f'{self.args.stage1}_back_translated_code']
                    
                    if pd.isna(original) or pd.isna(back_translated):
                        logger.warning(f"Missing code in row {idx + 1}, skipping...")
                        continue
                        
                    # Evaluate
                    metrics = self.evaluator.evaluate_code_pair(str(original), str(back_translated))
                    
                    # Store results
                    result = {
                        'chunk_size': chunk_size,
                        'index': idx + 1,
                        'original_code': original,
                        'translated_code': translated,
                        'back_translated_code': back_translated,
                        **metrics
                    }
                    results.append(result)
                    
                    logger.info(f"Evaluated translation {idx + 1} - Score: {metrics['overall_score']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating translation {idx + 1}: {str(e)}")
                    logger.error(f"Row content: {row.to_dict()}")
                    continue
                    
            if not results:
                logger.error("No translations were successfully evaluated")
                # Print first row for debugging
                if len(df) > 0:
                    logger.error("First row content:")
                    logger.error(df.iloc[0].to_dict())
                return False
                
            # Save results
            self.save_results(results, chunk_size)
            return True
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def save_results(self, results: List[Dict], chunk_size: int) -> None:
        """Save evaluation results."""
        try:
            # Save detailed results
            df = pd.DataFrame(results)
            details_file = self.eval_dir / \
                f"stage2/evaluation_details_{self.args.stage2}_{self.args.source_lang}_{self.args.target_lang}_{chunk_size}.csv" if not self.isStage1 else self.eval_dir/f"stage1/evaluation_details_{self.args.stage1}_{self.args.source_lang}_{self.args.target_lang}_{self.args.stage1_samples}.csv"
            df.to_csv(details_file, index=False)  
            
            # Calculate and save summary
            summary = {
                'chunk_size': chunk_size,
                'total_translations': len(results),
                'bleu_score': df['bleu_score'].mean(),
                'syntax_valid_rate': df['syntax_valid'].mean() * 100,
                'structure_score': df['structure_score'].mean(),
                'semantic_score': df['semantic_score'].mean(),
                'token_match': df['token_match'].mean(),
                'overall_score': df['overall_score'].mean()
            }
            
            summary_file = self.eval_dir / \
                f"stage2/evaluation_summary_{self.args.stage2}_{self.args.source_lang}_{self.args.target_lang}_{chunk_size}.json" if not self.isStage1 else self.eval_dir/f"stage1/evaluation_summary_{self.args.stage1}_{self.args.source_lang}_{self.args.target_lang}_{self.args.stage1_samples}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)
                
            self.log_summary(summary)
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise
    
    @staticmethod
    def log_summary(summary: Dict) -> None:
        """Log evaluation summary."""
        logger.info("\nEvaluation Summary:")
        logger.info(f"Chunk Size: {summary['chunk_size']}")
        logger.info(f"Total Translations: {summary['total_translations']}")
        logger.info(f"Average BLEU Score: {summary['bleu_score']:.4f}")
        logger.info(f"Syntax Valid Rate: {summary['syntax_valid_rate']:.1f}%")
        logger.info(f"Structure Score: {summary['structure_score']:.4f}")
        logger.info(f"Semantic Score: {summary['semantic_score']:.4f}")
        logger.info(f"Token Match: {summary['token_match']:.4f}")
        logger.info(f"Overall Score: {summary['overall_score']:.4f}")

def evaluate_translations(args, isStage1: bool = False) -> bool:
    """Main entry point for translation evaluation."""
    evaluator = TranslationEvaluator(args, isStage1)

    if not isStage1:
        chunk_sizes = [min(args.stage1_samples, size) for size in [0, 5, 10, 15, 25]]
        success = True
        
        for chunk_size in chunk_sizes:
            if not evaluator.evaluate_translations(chunk_size):
                logger.error(f"Evaluation failed for chunk size {chunk_size}")
                success = False
    else:
        success = True
        if not evaluator.evaluate_translations():
            logger.error("Evaluation failed for stage 1 translations")
            success = False
            
    return success