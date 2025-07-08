
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import traceback

# Add the current directory to Python path to import lx_anonymizer
current_dir = Path(__file__).parent.parent  # Go up to lx-anonymizer directory
sys.path.insert(0, str(current_dir))

try:
    from lx_anonymizer.video_reader import VideoReader
    from lx_anonymizer.custom_logger import logger, configure_global_logger
    # Try to import settings, but make it optional
    try:
        from lx_anonymizer.settings import DEFAULT_SETTINGS
    except ImportError:
        DEFAULT_SETTINGS = None
except ImportError as e:
    print(f"Error importing lx_anonymizer modules: {e}")
    print("Make sure you're running this from the lx-anonymizer directory.")
    print("Current working directory:", os.getcwd())
    print("Python path:", sys.path[:3])
    sys.exit(1)