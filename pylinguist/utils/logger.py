# pylinguist/utils/logger.py

import logging

def setup_logger(name: str = __name__) -> logging.Logger:
    """Set up logger with proper configuration to avoid duplicates."""
    
    # Get logger instance
    logger = logging.getLogger(name)
    
    # Only add handler if logger doesn't have handlers already
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Create and configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        # Prevent logging from propagating to the root logger
        logger.propagate = False
    
    return logger