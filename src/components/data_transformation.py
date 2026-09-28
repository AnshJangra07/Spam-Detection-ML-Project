import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OrdinalEncoder

from src.constant.training_pipeline import FEATURE_COLUMN, TARGET_COLUMN
from src.entity.artifact_entity import (
   DataIngestionArtifact,
   DataTransformationArtifact,
   DataValidationArtifact,
)
from src.entity.config_entity import DataTransformationConfig
from src.exception import SpamhamException
from src.logger import logging
from src.ml.model.text_preprocessor import preprocess_messages
from src.utils.main_utils import MainUtils


class DataTransformation:
   def __init__(
      self,
      data_ingestion_artifact: DataIngestionArtifact,
      data_validation_artifact: DataValidationArtifact,
      data_tranasformation_config: DataTransformationConfig,
   ):
      self.data_ingestion_artifact = data_ingestion_artifact
      self.data_validation_artifact = data_validation_artifact
      self.data_transformation_config = data_tranasformation_config
      self.utils = MainUtils()

   @staticmethod
   def read_data(file_path: str) -> pd.DataFrame:
      try:
         return pd.read_csv(file_path)
      except Exception as e:
         raise SpamhamException(e, sys) from e

   def get_stemmed_data(self, data: DataFrame) -> List[str]:
      try:
         messages = data[FEATURE_COLUMN].fillna("").astype(str)
         return preprocess_messages(messages)
      except Exception as e:
         raise SpamhamException(e, sys) from e

   def get_vectorized_data(
      self,
      train_df: DataFrame,
      test_df: DataFrame,
      vectorizer: CountVectorizer = CountVectorizer(),
   ) -> Tuple[np.ndarray, np.ndarray, CountVectorizer]:
      try:
         train_data = self.get_stemmed_data(train_df)
         test_data = self.get_stemmed_data(test_df)
         logging.info("Applying vectorizer object on training and testing data")
         vectorized_x_train = vectorizer.fit_transform(train_data)
         vectorized_x_test = vectorizer.transform(test_data)
         return vectorized_x_train.toarray(), vectorized_x_test.toarray(), vectorizer
      except Exception as e:
         raise SpamhamException(e, sys) from e

   def get_encoded_target_column(
      self,
      train_df: DataFrame,
      test_df: DataFrame,
      encoder: OrdinalEncoder = OrdinalEncoder(),
   ) -> Tuple[np.ndarray, np.ndarray, OrdinalEncoder]:
      try:
         y_train = train_df[[TARGET_COLUMN]]
         y_test = test_df[[TARGET_COLUMN]]
         encoded_y_train = encoder.fit_transform(y_train)
         encoded_y_test = encoder.transform(y_test)
         logging.info("Target columns have been encoded")
         return encoded_y_train, encoded_y_test, encoder
      except Exception as e:
         raise SpamhamException(e, sys) from e

   def initiate_data_transformation(self) -> DataTransformationArtifact:
      try:
         if not self.data_validation_artifact.validation_status:
            raise ValueError("Data validation failed; transformation was skipped")

         train_df = self.read_data(
            file_path=self.data_ingestion_artifact.trained_file_path
         )
         test_df = self.read_data(
            file_path=self.data_ingestion_artifact.test_file_path
         )
         x_train, x_test, vectorizer = self.get_vectorized_data(train_df, test_df)
         y_train, y_test, encoder = self.get_encoded_target_column(train_df, test_df)

         object_dir = os.path.dirname(
            self.data_transformation_config.transformed_vectorizer_object_file_path
         )
         os.makedirs(object_dir, exist_ok=True)
         self.utils.save_object(
            file_path=self.data_transformation_config.transformed_encoder_object_file_path,
            obj=encoder,
         )
         self.utils.save_object(
            file_path=self.data_transformation_config.transformed_vectorizer_object_file_path,
            obj=vectorizer,
         )

         train_array = np.c_[x_train, y_train]
         test_array = np.c_[x_test, y_test]
         self.utils.save_numpy_array_data(
            file_path=self.data_transformation_config.transformed_train_file_path,
            array=train_array,
         )
         self.utils.save_numpy_array_data(
            file_path=self.data_transformation_config.transformed_test_file_path,
            array=test_array,
         )

         artifact = DataTransformationArtifact(
            transformed_vectorizer_object_file_path=(
               self.data_transformation_config.transformed_vectorizer_object_file_path
            ),
            transformed_encoder_object_file_path=(
               self.data_transformation_config.transformed_encoder_object_file_path
            ),
            transformed_train_file_path=(
               self.data_transformation_config.transformed_train_file_path
            ),
            transformed_test_file_path=(
               self.data_transformation_config.transformed_test_file_path
            ),
         )
         logging.info("Data transformation artifact: %s", artifact)
         return artifact
      except Exception as e:
         raise SpamhamException(e, sys) from e
