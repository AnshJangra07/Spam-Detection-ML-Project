import os
from dotenv import load_dotenv
from pymongo import MongoClient
load_dotenv()

mongo_url = os.getenv("MONGODB_URL_KEY")
print("Mongo URL loaded:", bool(mongo_url))
client = MongoClient(mongo_url)
print("Connected:", client.admin.command("ping"))
db = client["spam_ham_database"]
print("Database:", db.name)
print("Collections:", db.list_collection_names())