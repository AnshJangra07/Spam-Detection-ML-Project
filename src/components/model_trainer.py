import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact

from src.exception import SpamhamException
from src.logger import logging
from src.utils.main_utils import MainUtils, load_numpy_array_data
from src.ml.model.estimator import SpamhamDetectionModel
from neuro_mf import ModelFactory




class ModelTrainer:
   def __init__(self, 
               data_transformation_artifact: DataTransformationArtifact,
               model_trainer_config: ModelTrainerConfig):
      
      self.data_transformation_artifact = data_transformation_artifact
      self.model_trainer_config = model_trainer_config
      self.utils = MainUtils()


   def initiate_model_trainer(self) -> ModelTrainerArtifact:
      logging.info("Entered initiate_model_trainer method of ModelTrainer class")

      try:
         train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
         test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
         x_train, y_train, x_test, y_test = train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1]
         y_test = np.asarray(y_test).ravel()
         
         
         model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
         best_model_detail = model_factory.get_best_model(X=x_train,y=y_train,base_accuracy=self.model_trainer_config.expected_f1_score)
         preprocessing_obj = self.utils.load_object(file_path=self.data_transformation_artifact.transformed_vectorizer_object_file_path)

         if best_model_detail.best_score < self.model_trainer_config.expected_f1_score:
            logging.info("No model reached the minimum expected F1 score")
            raise ValueError("No model reached the minimum expected F1 score")
         y_pred = np.asarray(best_model_detail.best_model.predict(x_test)).ravel()
            
         customer_segmentation_model = SpamhamDetectionModel(
               preprocessing_object=preprocessing_obj,
               trained_model_object=best_model_detail.best_model
         )
         logging.info("Spam Ham detection Model is created and saved.")
         trained_model_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
         os.makedirs(trained_model_path, exist_ok=True)
         
         self.utils.save_object(
               file_path=self.model_trainer_config.trained_model_file_path,
               obj=customer_segmentation_model
         )
         logging.info(f"Spam Ham detection Model is saved successfully at: {trained_model_path}")
         metric_artifact = ClassificationMetricArtifact(
            accuracy_score=accuracy_score(y_test, y_pred),
            f1_score=f1_score(y_test, y_pred, average="binary", zero_division=0),
            precision_score=precision_score(y_test, y_pred, average="binary", zero_division=0),
            recall_score=recall_score(y_test, y_pred, average="binary", zero_division=0),
         )
         logging.info(
            "Held-out metrics: accuracy=%.4f, precision=%.4f, recall=%.4f, f1=%.4f",
            metric_artifact.accuracy_score,
            metric_artifact.precision_score,
            metric_artifact.recall_score,
            metric_artifact.f1_score,
         )
         model_trainer_artifact = ModelTrainerArtifact(
         trained_model_file_path=self.model_trainer_config.trained_model_file_path,
         metric_artifact=metric_artifact,
         )

         logging.info("Model training completed successfully")
         logging.info(f"Model trainer artifact: {model_trainer_artifact}")

         return model_trainer_artifact

         

      except Exception as e:
         raise SpamhamException(e, sys) from e