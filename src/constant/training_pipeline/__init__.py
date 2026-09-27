# pipeline name and root directory constant
import os
from src.constant.s3_bucket import TRAINING_BUCKET_NAME


TARGET_COLUMN = "class"
FEATURE_COLUMN = "message"

PIPELINE_NAME: str = "src"
ARTIFACT_DIR: str = "artifact"
LOG_DIR = "logs"
LOG_FILE = "spamham.log"

# common file name

FILE_NAME: str = "spamham.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
VECTORIZER_OBJECT_FILE_NAME = "vectorizer.pkl"
ENCODER_OBJECT_FILE_NAME: str = "encoder.pkl"
MODEL_FILE_NAME = "model.pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = ""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2