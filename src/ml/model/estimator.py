import sys
from typing import Iterable

import numpy as np
from pandas import DataFrame, Series

from src.constant.training_pipeline import FEATURE_COLUMN
from src.exception import SpamhamException
from src.logger import logging
from src.ml.model.text_preprocessor import preprocess_messages


class SpamhamDetectionModel:
   def __init__(self, preprocessing_object: object, trained_model_object: object):
      self.preprocessing_object = preprocessing_object
      self.trained_model_object = trained_model_object

   def predict(self, dataframe: DataFrame | Series | Iterable[str]) -> np.ndarray:
      logging.info("Entered predict method of srcTruckModel class")

      try:
         logging.info("Using the trained model to get predictions")

         if isinstance(dataframe, DataFrame):
            if FEATURE_COLUMN in dataframe.columns:
               messages = dataframe[FEATURE_COLUMN].fillna("").astype(str).tolist()
            elif len(dataframe.columns) == 1:
               messages = dataframe.iloc[:, 0].fillna("").astype(str).tolist()
            else:
               raise ValueError(f"Input data must contain a '{FEATURE_COLUMN}' column")
         elif isinstance(dataframe, Series):
            messages = dataframe.fillna("").astype(str).tolist()
         else:
            messages = list(dataframe)

         transformed_feature = self.preprocessing_object.transform(
            preprocess_messages(messages)
         )

         logging.info("Used the trained model to get predictions")
         return self.trained_model_object.predict(transformed_feature)

      except Exception as e:
         raise SpamhamException(e, sys) from e

   def __repr__(self):
      return f"{type(self.trained_model_object).__name__}()"

   def __str__(self):
      return f"{type(self.trained_model_object).__name__}()"