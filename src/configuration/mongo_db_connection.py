import os
import sys
from dotenv import load_dotenv
load_dotenv()

import certifi
import pymongo

from src.constant.database import DATABASE_NAME
from src.constant.env_variable import MONGODB_URL_KEY

from src.exception import SpamhamException


ca = certifi.where()


class MongoDBClient:
   client = None

   def __init__(self, database_name=DATABASE_NAME) -> None:
      try:
         if MongoDBClient.client is None:
               mongo_db_url = os.getenv(MONGODB_URL_KEY)
               if mongo_db_url is None:
                  raise Exception(f"Environment key: {MONGODB_URL_KEY} is not set.")
               mongo_db_url = mongo_db_url.strip()
               if len(mongo_db_url) >= 2 and mongo_db_url[0] == mongo_db_url[-1] and mongo_db_url[0] in "\"'":
                  mongo_db_url = mongo_db_url[1:-1].strip()
               MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
         self.client = MongoDBClient.client
         self.database = self.client[database_name]
         self.database_name = database_name

      except Exception as e:
         raise SpamhamException(e, sys)

           
