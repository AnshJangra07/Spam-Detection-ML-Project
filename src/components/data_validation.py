import sys
from typing import Tuple

import pandas as pd
from pandas import DataFrame

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from src.entity.config_entity import DataValidationConfig

from src.exception import SpamhamException
from src.logger import logging
from src.utils.main_utils import MainUtils


class DataValidation:
   def __init__(
      self,
      data_ingestion_artifact: DataIngestionArtifact,
      data_validation_config: DataValidationConfig,
   ):
      self.data_ingestion_artifact = data_ingestion_artifact
      self.data_validation_config = data_validation_config

      self.utils = MainUtils()

      self._schema_config = self.utils.read_schema_config_file()

   def validate_schema_columns(self, dataframe: DataFrame) -> bool:
      """
      Method Name : validate_schema_columns
      Description : Validate the number of columns against the schema.

      Output      : True or False
      """

      try:
         required_columns = set(self._schema_config["columns"])
         actual_columns = list(dataframe.columns)
         status = (
            len(actual_columns) == len(required_columns)
            and set(actual_columns) == required_columns
         )

         logging.info(
            f"Required columns: {required_columns}; actual columns: {set(actual_columns)}; valid: {status}"
         )

         return status

      except Exception as e:
         raise SpamhamException(e, sys) from e

   def validate_dataset_schema_columns(
      self,
      train_set: DataFrame,
      test_set: DataFrame,
   ) -> Tuple[bool, bool]:
      """
      Method Name : validate_dataset_schema_columns
      Description : Validate schema columns for train and test datasets.

      Output      : Tuple containing train and test validation status.
      """

      logging.info(
         "Entered validate_dataset_schema_columns method "
         "of DataValidation class"
      )

      try:
         logging.info("Validating dataset schema columns")

         train_schema_status = self.validate_schema_columns(train_set)

         logging.info(
               "Validated dataset schema columns on the train set"
         )

         test_schema_status = self.validate_schema_columns(test_set)

         logging.info(
               "Validated dataset schema columns on the test set"
         )

         logging.info("Validated dataset schema columns")

         return train_schema_status, test_schema_status

      except Exception as e:
         raise SpamhamException(e, sys) from e

   @staticmethod
   def read_data(file_path) -> DataFrame:
      try:
         return pd.read_csv(file_path)

      except Exception as e:
         raise SpamhamException(e, sys) from e

   def initiate_data_validation(self) -> DataValidationArtifact:
      """
      Method Name : initiate_data_validation
      Description : Initiates the data validation component.
      """

      logging.info(
         "Entered initiate_data_validation method of DataValidation class"
      )

      try:
         logging.info("Initiated data validation for the dataset")

         train_df = DataValidation.read_data(
               file_path=self.data_ingestion_artifact.trained_file_path
         )

         test_df = DataValidation.read_data(
               file_path=self.data_ingestion_artifact.test_file_path
         )

         (
               schema_train_col_status,
               schema_test_col_status,
         ) = self.validate_dataset_schema_columns(
               train_set=train_df,
               test_set=test_df,
         )

         logging.info(
               f"Schema train cols status is "
               f"{schema_train_col_status} and "
               f"schema test cols status is "
               f"{schema_test_col_status}"
         )

         if schema_train_col_status and schema_test_col_status:
               logging.info(
                  "Dataset schema validation completed"
               )
               validation_status = True
         else:
               logging.warning(
                  "Dataset schema validation failed"
               )
               validation_status = False

         data_validation_artifact = DataValidationArtifact(
               validation_status=validation_status,
               valid_train_file_path=(
                  self.data_ingestion_artifact.trained_file_path
               ),
               valid_test_file_path=(
                  self.data_ingestion_artifact.test_file_path
               ),
               invalid_train_file_path=(
                  self.data_validation_config.invalid_train_file_path
               ),
               invalid_test_file_path=(
                  self.data_validation_config.invalid_test_file_path
               ),
               drift_report_file_path=(
                  self.data_validation_config.drift_report_file_path
               ),
         )

         return data_validation_artifact

      except Exception as e:
         raise SpamhamException(e, sys) from e