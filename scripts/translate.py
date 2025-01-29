import argparse
from pathlib import Path
import sys
from data.input.preprocessor import DatasetPreprocessor 
from pylinguist.utils.partial_translator import partial_translate_examples
from pylinguist.utils.logger import setup_logger
from pylinguist.models.stage1.google import GoogleTranslator
from pylinguist.models.stage1.deepl import DeepLTranslator
import pandas as pd 
from tqdm import tqdm

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
    parser.add_argument('--stage1', type=str, choices=['google', 'deepl'], required=True)
    parser.add_argument('--stage2', type=str, choices=['gpt', 'llama', 'claude'])
    return parser.parse_args()

def check_paths():
    """Create necessary directories."""
    paths = [
        Path("data/input/samples"),
        Path("data/output/partial_translation"),
        Path("data/output/stage1"),
        Path("data/output/stage2")
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
        elif args.stage1 == 'deepl':
            translator = DeepLTranslator(source_lang=args.source_lang, target_lang=args.target_lang)
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
        elif args.stage2 == 'llama':
            logger.error("Llama translator not implemented yet")
            return False
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

def run_back_translation(args, stage2_files, chunk_list):

    print("stage2_files", stage2_files) ## stage2_files is a dataframe
    print("chunk_list", chunk_list) # single digit number

    try:
        # Use stage2 dataframe and First do the partially translate the f'{stage2}_translated_code using joshua_keyword
        logger.info(f"Step 1: Starting partial back translation using Joshua keyword...")
        results = partial_translate_examples(
            data_path=stage2_files,
            source_lang=args.target_lang,
            target_lang=args.source_lang,
            start_index=0,
            stage1_samples=0,
            stage2_samples=args.stage2_samples,
            back_translation=True,
            stage2_model=args.stage2
        )

        output_dir = Path("data/output/back_translation/partial_back_translation")
        output_file = output_dir / f"partial_back_translation_{args.stage2}_{args.source_lang}_{args.target_lang}_{args.stage2_samples}_{chunk_list}.csv"
        results.to_csv(output_file, index=False)
        logger.info(f"Partial back translation saved to: {output_file}")

        return True


    except Exception as e:
        logger.error(f"Back translation error: {str(e)}")
        return False

    
def main():
    args = parse_args()
    check_paths()
    
    try:
        # Preprocessing
        if not check_dataset_exists() or args.force_preprocess:
            if not run_preprocessing(args):
                return 1
        else:
            logger.info("Dataset exists. Skipping preprocessing.")
            
        if args.preprocess_only:
            return 0
            
        # Partial Translation
        partial_file = Path("data/output/partial_translation") / \
            f"partial_translation_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.stage1_samples}_{args.stage2_samples}.csv"
            
        if not partial_file.exists():
            if not run_partial_translation(args):
                return 1
        else:
            logger.info("Partial translation already exists. Skipping.")
                
        # Stage 1
        stage1_file = Path("data/output/stage1") / \
            f"Stage_1_{args.stage1}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.stage1_samples}_{args.stage2_samples}.csv"
            
        if not stage1_file.exists():
            if not run_stage1_translation(args, pd.read_csv(partial_file)):
                return 1
        else:
            logger.info("Stage 1 translation already exists. Skipping.")
                
        if not args.stage2:
            logger.info("Stage 1 pipeline completed successfully")
            return 0
            
        # Stage 2
        chunk_sizes = [min(args.stage1_samples, size) for size in [5, 10, 15, 25]]
        stage2_files = [
            Path("data/output/stage2") / \
            f"Stage_2_{args.stage2}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.stage1_samples}_{args.stage2_samples}_{chunk_size}.csv"
            for chunk_size in chunk_sizes
        ]
        
        missing_chunks = [chunk_sizes[i] for i, file in enumerate(stage2_files) if not file.exists()]
        
        if missing_chunks:
            if not run_stage2_translation(args, pd.read_csv(stage1_file), pd.read_csv(partial_file), missing_chunks):
                return 1
        else:
            logger.info("Stage 2 translation already exists. Skipping.")
            
        logger.info("Forward translation pipeline completed successfully")

      # Back Translation section in main()
        logger.info("Starting back translation process...")
        
        # Create necessary directories
        Path("data/output/back_translation/partial_back_translation").mkdir(parents=True, exist_ok=True)
        
        # Process each stage2 file for back translation
        for chunk_size in chunk_sizes:
            # Check if stage2 file exists for this chunk size
            stage2_file = Path("data/output/stage2") / \
                f"Stage_2_{args.stage2}_{args.source_lang}_{args.target_lang}_{args.start_index}_{args.stage1_samples}_{args.stage2_samples}_{chunk_size}.csv"
            
            if not stage2_file.exists():
                logger.warning(f"Stage 2 file not found for chunk size {chunk_size}, skipping...")
                continue
                
            # Check if partial back translation already exists
            partial_back_file = Path("data/output/back_translation/partial_back_translation/") / \
                f"partial_back_translation_{args.stage2}_{args.source_lang}_{args.target_lang}_{args.stage2_samples}_{chunk_size}.csv"
            
            if partial_back_file.exists():
                logger.info(f"Partial back translation already exists for chunk size {chunk_size}, skipping...")
                continue
                
            logger.info(f"Processing back translation for chunk size {chunk_size}")
            
            # Run back translation for this chunk
            if not run_back_translation(args, stage2_file, chunk_size):
                logger.error(f"Back translation failed for chunk size {chunk_size}")
                return 1
            
        logger.info("Back translation process completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
if __name__ == "__main__":
    sys.exit(main())