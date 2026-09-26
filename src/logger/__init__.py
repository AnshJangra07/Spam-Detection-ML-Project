# logger_setup.py

import logging
import os
from src.constant.training_pipeline import ARTIFACT_DIR, LOG_DIR, LOG_FILE, PIPELINE_NAME

# Assumes this file is at root level
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
logs_path = os.path.join(CURRENT_DIR, PIPELINE_NAME, ARTIFACT_DIR, LOG_DIR)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)
