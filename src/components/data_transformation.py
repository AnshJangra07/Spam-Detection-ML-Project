import sys
from datetime import datetime
import numpy as np
import os
import pandas as pd

from imblearn.combine import SMOTETomek
from pandas import DataFrame
import re

#nltk modules
import nltk
from typing import List, Union 
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#sklearn modules
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OrdinalEncoder

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.components.data_ingestion import DataIngestion

from src.constant.training_pipeline import *
from src.entity.config_entity import SimpleImputerConfig
from src.exception import SpamhamException
from src.logger import logging
from src.utils.main_utils import MainUtils

import warnings
warnings.filterwarnings('ignore')


class DataTransformation:
   def __init__(self,
               data_ingestion_artifact:DataIngestionArtifact,
               data_validation_artifact: DataValidationArtifact,
               data_tranasformation_config: DataTransformationConfig):
      
      self.data_ingestion_artifact = data_ingestion_artifact
      self.data_validation_artifact = data_validation_artifact
      self.data_transformation_config = data_tranasformation_config
      self.data_ingestion = DataIngestion()

      self.imputer_config = SimpleImputerConfig()

      self.utils = MainUtils()
      
      
      
   
   @staticmethod
   def read_data(file_path:str) -> pd.DataFrame:
      try:
         return pd.read_csv(file_path)
      except Exception as e:
         raise SpamhamException(e,sys)
      
      
   


   

   def get_stemmed_data(self, data: DataFrame, stemmer = PorterStemmer()) -> List:
      
      try:
         corpus = []
         
         for index in range(0, len(data)):
               words = re.sub('[^a-zA-Z]', ' ', data[FEATURE_COLUMN][index])
               words = words.lower()
               words = words.split()
               
               words = [stemmer.stem(word) for word in words if not word in stopwords.words('english')]
               words = ' '.join(words)
               corpus.append(words)
               
         return corpus
      
      except Exception as e:
         raise SpamhamException(e, sys)
         
      
   def get_vectorized_data(self, train_df:DataFrame, test_df:DataFrame, vectorizer = CountVectorizer()) -> Union[np.ndarray, np.ndarray, object]:
      
      try:
         train_data = self.get_stemmed_data(train_df)
         test_data = self.get_stemmed_data(test_df)
         
         logging.info(f"Applying vectorizer object on training dataframe and testing dataframe")

         vectorized_x_train = vectorizer.fit_transform(train_data)
         vectorized_x_test = vectorizer.transform(test_data)
         
         logging.info("x_train and x_test have been vectorized.")
         
         x_train = vectorized_x_train.toarray()
         x_test = vectorized_x_test.toarray()
         
         
         return x_train, x_test, vectorizer
      
      except Exception as e:
         raise SpamhamException(e, sys)
      
      
   def get_encoded_target_column(self, train_df:DataFrame, test_df:DataFrame, encoder = OrdinalEncoder()) -> Union[np.ndarray, np.ndarray, object]:
      
      try:
         y_train = train_df[[TARGET_COLUMN]]
         y_test = test_df[[TARGET_COLUMN]]
         
         encoded_y_train = encoder.fit_transform(y_train)
         encoded_y_test =  encoder.transform(y_test)
         
         logging.info("target columns has been encoded.")
         
         return encoded_y_train, encoded_y_test, encoder
      
      except Exception as e:
         raise SpamhamException(e, sys)
      
      
      
   
   def initiate_data_transformation(self)->DataTransformationArtifact:
      try:
         if self.data_validation_artifact.validation_status:
            
               
               train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
               test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)
               
            
               
               X_train, X_test, vectorizer = self.get_vectorized_data(train_df=train_df, test_df=test_df)
               y_train, y_test, encoder = self.get_encoded_target_column(train_df=train_df, test_df=test_df)
               
               
               #creating preprocessing obj directory
               preprocessor_obj_dir = os.path.dirname(self.data_transformation_config.transformed_vectorizer_object_file_path)
               os.makedirs(preprocessor_obj_dir, exist_ok=True)
               
               #dump encoder output
               encoder_file_path = self.data_transformation_config.transformed_encoder_object_file_path
               self.utils.save_object(file_path=encoder_file_path,obj=encoder)
               logging.info("Encoder object saved to: %s" % encoder_file_path)
               
               #dump vectorizer object
               vectorizer_file_path = self.data_transformation_config.transformed_vectorizer_object_file_path
               logging.info(f"Saving vectorizer object.")
               self.utils.save_object(file_path=vectorizer_file_path,obj=vectorizer)
         
               

               #concatinating input features and targets
               train_arr = np.c_[X_train, np.array(y_train)]

               test_arr = np.c_[X_test, np.array(y_test)]
                              

               logging.info(f"Saving transformed training and testing array.")
               
               self.utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_file_path,array=train_arr)
               self.utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_file_path,array=test_arr)

               
               data_transformation_artifact = DataTransformationArtifact(
                  transformed_vectorizer_object_file_path=self.data_transformation_config.transformed_vectorizer_object_file_path,
                  transformed_encoder_object_file_path= self.data_transformation_config.transformed_encoder_object_file_path,
                  transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                  transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
               )
               logging.info(f"Data transformationa artifact: {data_transformation_artifact}")

               return data_transformation_artifact
      except Exception as e:
         raise SpamhamException(e,sys) from e