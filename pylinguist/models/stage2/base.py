from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pandas as pd
from ...utils.logger import setup_logger

logger = setup_logger('pylinguist.stage2.base')

class BaseEnhancer(ABC):
    def __init__(self, source_lang: str, target_lang: str, **kwargs):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.example_count = kwargs.get('example_count', 5)
        self.batch_size = kwargs.get('batch_size', 5)
        self.cache = {}
        
    def load_examples(self, examples_file: str) -> pd.DataFrame:
        """Load example translation pairs for few-shot learning."""
        try:
            df = pd.read_csv(examples_file)
            self.examples = df.head(self.example_count)[['English_code', f'{self.target_lang}_code']]
            return self.examples
        except Exception as e:
            logger.error(f"Error loading examples: {str(e)}")
            return pd.DataFrame()
            
    def create_prompt(self, code_to_translate: str) -> str:
        """Create prompt with examples and code to translate."""
        examples_text = ""
        for i, row in self.examples.iterrows():
            examples_text += f"\n\nExample {i+1}:\n"
            examples_text += f"Source code:\n{row['English_code']}\n"
            examples_text += f"Translated code:\n{row[f'{self.target_lang}_code']}\n"
            examples_text += "------------------------\n"
            
        prompt = f"""Complete the translation of this Python code to {self.target_lang}:
        - Translate variable names, function names, strings and comments
        - Join multi-word translations with underscores
        - Break down compound words separated by underscores and translate each part
        - Preserve code structure and syntax
        - Examples of translations:
    
        {examples_text}
        
        Now translate this code to {self.target_lang}:
        {code_to_translate}"""

        return prompt
        
    @abstractmethod
    def enhance_translation(self, code: str) -> str:
        """Enhance the translation using the model."""
        pass

    def process_batch(self, codes: List[str]) -> List[str]:
        """Process a batch of code translations."""
        enhanced_codes = []
        for code in codes:
            if code in self.cache:
                enhanced_codes.append(self.cache[code])
                continue
                
            enhanced = self.enhance_translation(code)
            self.cache[code] = enhanced
            enhanced_codes.append(enhanced)
            
        return enhanced_codes

    def run_enhancement(self, input_file: str, output_file: str) -> pd.DataFrame:
        """Run enhancement pipeline on a dataset."""
        try:
            # Load and split data
            df = pd.read_csv(input_file)
            test_data = df.iloc[self.example_count:]
            
            # Process translations
            enhanced_translations = []
            for i in range(0, len(test_data), self.batch_size):
                batch = test_data.iloc[i:i + self.batch_size]
                enhanced_batch = self.process_batch(batch['stage1_translated_code'].tolist())
                enhanced_translations.extend(enhanced_batch)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'source_code': test_data['source_code'].tolist(),
                'stage1_translated_code': test_data['stage1_translated_code'].tolist(),
                'stage2_translated_code': enhanced_translations
            })
            
            # Save results
            results.to_csv(output_file, index=False)
            return results
            
        except Exception as e:
            logger.error(f"Enhancement pipeline error: {str(e)}")
            return pd.DataFrame()