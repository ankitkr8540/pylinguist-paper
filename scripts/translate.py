import argparse
from pathlib import Path
import sys
from typing import List
from data.input.preprocessor import DatasetPreprocessor
from pylinguist.utils.partial_translator import partial_translate_examples
from pylinguist.utils.logger import setup_logger
from pylinguist.models.stage1.google import GoogleTranslator
from evaluate import evaluate_translations
import pandas as pd 
from tqdm import tqdm
import traceback
import json

logger = setup_logger()

def parse_args():
    parser = argparse.ArgumentParser(description='PyLinguist - Code Translation Pipeline')
    parser.add_argument('--preprocess-only', action='store_true')
    parser.add_argument('--dataset', type=str, default="jtatman/python-code-dataset-500k")
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--force-preprocess', action='store_true')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--stage1-samples', type=int, default=10)
    parser.add_argument('--stage2-samples', type=int, default=30)
    parser.add_argument('--source-lang', type=str, required=True)
    parser.add_argument('--target-lang', type=str, required=True)
    parser.add_argument('--stage1', type=str, choices=['google'], required=True)
    parser.add_argument('--stage2', type=str, choices=['gpt', 'claude'])
    return parser.parse_args()

def check_paths():
    """Create necessary directories."""
    paths = [
        Path("data/input/samples"),
        Path("data/output/partial_translation"),
        Path("data/output/stage1"),
        Path("data/output/stage2"),
        Path("data/output/back_translation/stage1/partial_back_translation"),
        Path("data/output/back_translation/stage1/final_back_translation"),
        Path("data/output/back_translation/stage2/partial_back_translation"),
        Path("data/output/back_translation/stage2/final_back_translation"),
        Path("data/output/evaluation/stage1"),
        Path("data/output/evaluation/stage2"),
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

def check_dataset_exists():
    return Path("data/input/samples/python_code_dataset.csv").exists()

def run_preprocessing(args):
    try:
        logger.info("Starting preprocessing pipeline...")
        preprocessor = DatasetPreprocessor("data/input/samples")
        results = preprocessor.process_and_save_dataset(args)
        
        if not results['success']:
            logger.error(f"Preprocessing failed: {results.get('error')}")
            return False
            
        logger.info(f"Dataset saved to: {results['output_path']}")
        return True
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        return False

def run_partial_translation(args):
    try:
        logger.info("Starting partial translation...")
        dataset_path = Path("data/input/samples/python_code_dataset.csv")
        
        results = partial_translate_examples(
            data_path=dataset_path,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            start_index=args.start_index,
            stage1_samples=args.stage1_samples,
            stage2_samples=args.stage2_samples
        )
        
        output_dir = Path("data/output/partial_translation")
        output_file = output_dir / f"partial_translation_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.stage1_samples}_{args.stage2_samples}.csv"
        results.to_csv(output_file, index=False)
        
        logger.info(f"Partial translation saved to: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Partial translation error: {str(e)}")
        return False

def run_stage1_translation(args, partial_df):
    try:
        logger.info(f"Starting Stage 1 translation using {args.stage1}...")
        
        if args.stage1 == 'google':
            translator = GoogleTranslator(source_lang=args.source_lang, target_lang=args.target_lang)
        else:
            logger.error("Invalid Stage 1 translator")
            return False
            
        translated_lines = []
        for i, row in tqdm(partial_df.iterrows(), total=args.stage1_samples):
            if i >= args.stage1_samples:
                break
            translated_code = translator.translate_code(row['Partial_translated_code'])
            translated_lines.append({
                'English_code': row['English_code'],
                'Partial_translated_code': row['Partial_translated_code'],
                f'{args.stage1}_translated_code': translated_code
            })
            
        output_dir = Path("data/output/stage1")
        output_file = output_dir / f"Stage_1_{args.stage1}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.stage1_samples}_{args.stage2_samples}.csv"
        pd.DataFrame(translated_lines).to_csv(output_file, index=False)
        
        logger.info(f"Stage 1 translation saved to: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Stage 1 translation error: {str(e)}")
        return False

def run_stage2_translation(args, stage1_df, partial_df, chunk_list):
    """Run Stage 2 translation using selected model."""
    try:
        logger.info(f"Starting Stage 2 translation using {args.stage2}...")
        
        # Initialize appropriate model
        if args.stage2 == 'gpt':
            from pylinguist.models.stage2.gpt import GPTEnhancer
            enhancer = GPTEnhancer(
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                translator_name=args.stage1
            )
        elif args.stage2 == 'claude':
            from pylinguist.models.stage2.claude import ClaudeTranslator
            enhancer = ClaudeTranslator(
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                translator_name=args.stage1
            )

            
        else:
            logger.error("Invalid Stage 2 translator")
            return False
        
        # Get stage2 samples from partial translations
        translated_lines = []
        start_idx = args.start_index + args.stage1_samples
        end_idx = start_idx + args.stage2_samples
        
        # Process each sample
        for chunk_size in chunk_list:
            for i, row in tqdm(partial_df.iloc[start_idx:end_idx].iterrows(), 
                            total=args.stage2_samples, 
                            desc="Stage 2 Translation with few shot Example Size of " + str(chunk_size)):
                translated_code = enhancer.enhance_translation(
                    code=row['Partial_translated_code'],
                    examples_df=stage1_df[args.start_index:chunk_size if chunk_size <= len(stage1_df) else len(stage1_df)]
                )
                
                translated_lines.append({
                    'English_code': row['English_code'],
                    'Partial_translated_code': row['Partial_translated_code'],
                    f'{args.stage2}_translated_code': translated_code
                })
            
            # Save results
            output_dir = Path("data/output/stage2")
            output_file = output_dir / f"Stage_2_{args.stage2}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.stage1_samples}_{args.stage2_samples}_{chunk_size}.csv"
            pd.DataFrame(translated_lines).to_csv(output_file, index=False)
        logger.info(f"Stage 2 translation saved to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Stage 2 translation error: {str(e)}")
        return False

def run_partial_back_translation(args, stage2_files, chunk_list, is_stage1=False):

    try:
        # Use stage2 dataframe and First do the partially translate the f'{stage2}_translated_code using unipy_keyword
        results = partial_translate_examples(
            data_path=stage2_files,
            source_lang=args.target_lang,
            target_lang=args.source_lang,
            start_index=0,
            stage1_samples=0,
            stage2_samples=args.stage2_samples if not is_stage1 else args.stage1_samples,
            back_translation=True,
            stage2_model=args.stage2 if not is_stage1 else args.stage1
        )

        if not is_stage1:
            output_dir = Path("data/output/back_translation/stage2/partial_back_translation")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"partial_back_translation_{args.stage2}_{args.source_lang}_{args.target_lang}_{args.stage2_samples}_{chunk_list}.csv"
            results.to_csv(output_file, index=False)
            logger.info(f"Stage 2 Partial back translation saved to: {output_file}")
        else:
            output_dir = Path("data/output/back_translation/stage1/partial_back_translation")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"partial_back_translation_{args.stage1}_{args.source_lang}_{args.target_lang}_{args.stage1_samples}.csv"
            results.to_csv(output_file, index=False)
            logger.info(f"Stage 1 partial back translation saved to: {output_file}")

        return True


    except Exception as e:
        logger.error(f"Back translation error: {str(e)}")
        return False

def run_final_back_translation(args, stage1_df, partial_df, chunk):
    try:
        logger.info(f"Step 2: Starting final back translation using {args.stage2}...")
        
        # Initialize appropriate model
        if args.stage2 == 'gpt':
            from pylinguist.models.stage2.gpt import GPTEnhancer
            enhancer = GPTEnhancer(
                source_lang=args.target_lang,
                target_lang=args.source_lang,
                translator_name=args.stage1
            )
        elif args.stage2 == 'claude':
            from pylinguist.models.stage2.claude import ClaudeTranslator
            enhancer = ClaudeTranslator(
                source_lang=args.target_lang,
                target_lang=args.source_lang,
                translator_name=args.stage1
            )
            
        else:
            logger.error("Invalid Stage 2 translator")
            return False
        
        # Get stage2 samples from partial translations
        translated_lines = []
        start_idx = args.start_index + args.stage1_samples
        end_idx = start_idx + args.stage2_samples
        
        # Process each sample
        for i, row in tqdm(partial_df.iterrows(), total=args.stage2_samples):
            translated_code = enhancer.enhance_translation(
                code = row[f'{args.stage2}_partial_back_translated_code'],
                examples_df = stage1_df[args.start_index:chunk if chunk <= len(stage1_df) else len(stage1_df)]
            )

            translated_lines.append({
                'Original Code': row['English_code'],
                f'{args.stage2}_partial_translated_code': row[f'{args.stage2}_partial_back_translated_code'],
                f'{args.stage2}_back_translated_code': translated_code
            })

        # Save results
        output_dir = Path("data/output/back_translation/stage2/final_back_translation")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"Final_back_translation_{args.stage2}_{args.source_lang}_{args.target_lang}_{args.stage2_samples}_{chunk}.csv"
        pd.DataFrame(translated_lines).to_csv(output_file, index=False)

        logger.info(f"Final back translation saved to: {output_file}")
        return True
    
    except Exception as e:
        logger.error(f"Final back translation error: {str(e)}")
        return False

def run_stage1_final_back_translation(args, partial_df):
    try:
        logger.info(f"Starting Stage 1 back translation using {args.stage1}...")
        
        if args.stage1 == 'google':
            translator = GoogleTranslator(source_lang=args.target_lang, target_lang=args.source_lang)
        else:
            logger.error("Invalid Stage 1 translator")
            return False
            
        translated_lines = []
        for i, row in tqdm(partial_df.iterrows(), total=args.stage1_samples):
            if i >= args.stage1_samples:
                break
            translated_code = translator.translate_code(row[f'{args.stage1}_partial_back_translated_code'])
            translated_lines.append({
                'English_code': row['English_code'],
                'Partial_translated_code': row[f'{args.stage1}_partial_back_translated_code'],
                f'{args.stage1}_back_translated_code': translated_code
            })
            
        output_dir = Path("data/output/back_translation/stage1/final_back_translation")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"final_back_translation_{args.stage1}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.stage1_samples}_{args.stage2_samples}.csv"
        pd.DataFrame(translated_lines).to_csv(output_file, index=False)
        
        logger.info(f"Stage 1 back translation saved to: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Stage 1 back translation error: {str(e)}")
        return False

def run_stage1_evaluation(args, stage1_df):
    try:
        logger.info("Starting evaluation for Stage 1 translation...")
        partial_dir = Path("data/output/back_translation/stage1/partial_back_translation")
        partial_file = partial_dir / f"partial_back_translation_{args.stage1}_{args.source_lang}_{args.target_lang}_{args.stage1_samples}.csv"
        if not partial_file.exists():
            logger.info("Running partial back translation for evaluation...")
            if run_partial_back_translation(args, stage1_df, 0, is_stage1=True):
                logger.info("Partial back translation saved successfully")
            else:
                logger.error("Partial back translation failed")
                return False
        else:
            logger.info("Partial back translation exists. Skipping.")
        stage1_back_df = pd.read_csv(partial_file)
        final_dir = Path("data/output/back_translation/stage1/final_back_translation")
        final_file = final_dir / f"final_back_translation_{args.stage1}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.stage1_samples}_{args.stage2_samples}.csv"
        if not final_file.exists():
            logger.info("Running final back translation for evaluation...")
            if run_stage1_final_back_translation(args, stage1_back_df):
                logger.info("Final back translation saved successfully")
            else:
                logger.error("Final back translation failed")
                return False
        else:
            logger.info("Final back translation exists. Skipping.")
        eval_dir = Path("data/output/evaluation/stage1")
        eval_file = eval_dir / f"stage1/evaluation_details_{args.stage1}_{args.source_lang}_{args.target_lang}_{args.stage1_samples}.csv"
        if not eval_file.exists():
            if evaluate_translations(args, isStage1=True):
                logger.info("Evaluation completed successfully")
                return True
            else:
                logger.error("Evaluation failed")
                return False
        else:
            logger.info("Evaluation file exists. Skipping.")
    except Exception as e:
        logger.error(f"Stage 1 evaluation error: {str(e)}")
        return False
    
def main():
    args = parse_args()
    check_paths()
    
    try:
        # Preprocessing
        logger.info("Starting preprocessing check...")
        if not check_dataset_exists() or args.force_preprocess:
            logger.info("Running preprocessing pipeline...")
            if not run_preprocessing(args):
                logger.error("Preprocessing failed")
                return 1
        else:
            logger.info("Dataset exists. Skipping preprocessing.")
            
        if args.preprocess_only:
            logger.info("Preprocess only flag set. Exiting.")
            return 0
            
        # Forward Translation Pipeline
        logger.info(f"Starting translation pipeline: {args.source_lang} -> {args.target_lang}")
            
        # Partial Translation
        partial_file = Path("data/output/partial_translation") / \
            f"partial_translation_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.stage1_samples}_{args.stage2_samples}.csv"
            
        if not partial_file.exists():
            logger.info("Starting partial translation...")
            if not run_partial_translation(args):
                logger.error("Partial translation failed")
                return 1
        else:
            logger.info("Partial translation exists. Skipping.")
                
        # Stage 1 Translation
        stage1_file = Path("data/output/stage1") / \
            f"Stage_1_{args.stage1}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.stage1_samples}_{args.stage2_samples}.csv"
            
        if not stage1_file.exists():
            logger.info(f"Running Stage 1 translation using {args.stage1}...")
            try:
                partial_df = pd.read_csv(partial_file)
                if not run_stage1_translation(args, partial_df):
                    logger.error("Stage 1 translation failed")
                    return 1
            except Exception as e:
                logger.error(f"Error reading partial translation file: {str(e)}")
                return 1
        else:
            logger.info("Stage 1 translation exists. Skipping.")

        # Stage 1 Evaluation
        stage1_eval = Path("data/output/evaluation/stage1") / \
            f"Stage_1_{args.stage1}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.stage1_samples}_{args.stage2_samples}_eval.csv"

        if not stage1_eval.exists():
            logger.info(f"Running evaluation for Stage 1 translation using {args.stage1}...")
            try:
                # stage1_df = pd.read_csv(stage1_file)
                if not run_stage1_evaluation(args, stage1_file):
                    logger.error("Stage 1 evaluation failed")
                    return 1
            except Exception as e:
                logger.error(f"Error reading Stage 1 translation file: {str(e)}")
                return 1
            
                
        if not args.stage2:
            logger.info("Stage 1 pipeline completed successfully")
            return 0
            
        # Stage 2 Translation
        base_chunk_sizes = [0,5, 10, 15, 25]
        chunk_sizes = list(set([min(args.stage1_samples, size) for size in base_chunk_sizes]))
        chunk_sizes.sort()  # Ensure ordered processing
        logger.info(f"Stage 2: Processing chunk sizes: {chunk_sizes}")
        
        stage2_files = {
            chunk_size: Path("data/output/stage2") / \
                f"Stage_2_{args.stage2}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.stage1_samples}_{args.stage2_samples}_{chunk_size}.csv"
            for chunk_size in chunk_sizes
        }
        
        missing_chunks = [size for size, file in stage2_files.items() if not file.exists()]
        
        if missing_chunks:
            logger.info(f"Processing missing Stage 2 chunks: {missing_chunks}")
            try:
                stage1_df = pd.read_csv(stage1_file)
                partial_df = pd.read_csv(partial_file)
                if not run_stage2_translation(args, stage1_df, partial_df, missing_chunks):
                    logger.error("Stage 2 translation failed")
                    return 1
                for chunk_size in missing_chunks:
                    logger.info(f"Stage 2 file saved: {stage2_files[chunk_size]}")
            except Exception as e:
                logger.error(f"Error in Stage 2 translation: {str(e)}")
                return 1
        else:
            logger.info("All Stage 2 translations exist. Skipping.")
            
        logger.info("Forward translation pipeline completed")

        # Back Translation Pipeline
        logger.info(f"Starting back translation pipeline: {args.target_lang} -> {args.source_lang}")
        
        # Create back translation directories
        back_translation_dirs = {
            'partial': Path("data/output/back_translation/stage2/partial_back_translation"),
            'final': Path("data/output/back_translation/stage2/final_back_translation")
        }
        
        for dir_path in back_translation_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Process each chunk size for back translation
        for chunk_size in chunk_sizes:
            logger.info(f"\nProcessing back translation for chunk size {chunk_size}")
            
            # Define files for this chunk
            stage2_file = stage2_files[chunk_size]
            partial_back_file = back_translation_dirs['partial'] / \
                f"partial_back_translation_{args.stage2}_{args.source_lang}_{args.target_lang}_{args.stage2_samples}_{chunk_size}.csv"
            final_back_file = back_translation_dirs['final'] / \
                f"Final_back_translation_{args.stage2}_{args.source_lang}_{args.target_lang}_{args.stage2_samples}_{chunk_size}.csv"
            
            # Skip if already processed
            if partial_back_file.exists() and final_back_file.exists():
                logger.info(f"Back translation exists for chunk size {chunk_size}. Skipping.")
                continue
                
            try:
                # Step 1: Partial Back Translation
                if not partial_back_file.exists():
                    logger.info("Running partial back translation...")
                    if not run_partial_back_translation(args, stage2_file, chunk_size):
                        logger.error("Partial back translation failed")
                        return 1
                    logger.info(f"Partial back translation saved: {partial_back_file}")
                
                # Step 2: Final Back Translation

                if not final_back_file.exists():
                    logger.info("Running final back translation...")
                    stage1_df = pd.read_csv(stage1_file)
                    partial_back_df = pd.read_csv(partial_back_file)
                    
                    if not run_final_back_translation(args, stage1_df, partial_back_df, chunk_size):
                        logger.error("Final back translation failed")
                        return 1
                    logger.info(f"Final back translation saved: {final_back_file}")
                
            except Exception as e:
                logger.error(f"Error in back translation for chunk {chunk_size}: {str(e)}")
                logger.error(traceback.format_exc())
                return 1
            
        logger.info("\nComplete translation pipeline finished successfully")
      # Run evaluation
        if evaluate_translations(args):
            logger.info("Evaluation completed successfully")
        else:
            logger.error("Evaluation failed")
            
        logger.info("\nComplete pipeline with evaluation finished successfully")
        return 0

    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    
if __name__ == "__main__":
    sys.exit(main())