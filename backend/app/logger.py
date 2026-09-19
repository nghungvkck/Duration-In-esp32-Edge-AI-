"""
logger.py: cấu hình logging cho backend
"""

import logging
import sys
from pathlib import Path

#====================================
# Path
#====================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / 'backend.log'
ERROR_FILE = LOG_DIR / 'backend_error.log'

# ============================================================
# FORMATTERS
# ============================================================
FILE_FORMAT = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
CONSOLE_FORMAT = logging.Formatter(
    '%(levelname)-8s | %(name)s | %(message)s'
)


# ============================================================
# SETUP
# ============================================================
def setup_logger(name: str = 'app') -> logging.Logger:
    """Tạo logger với file + console handler."""
    logger = logging.getLogger(name)
    
    # Tránh add handler trùng khi gọi nhiều lần
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # --- File handler (all logs) ---
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(FILE_FORMAT)
    
    # --- File handler (chỉ ERROR trở lên) ---
    error_handler = logging.FileHandler(ERROR_FILE, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(FILE_FORMAT)
    
    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(CONSOLE_FORMAT)
    
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)
    
    return logger


# Logger dùng chung
logger = setup_logger('app')