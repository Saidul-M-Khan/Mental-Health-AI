from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME")

if not DATABASE_NAME:
    raise ValueError("DATABASE_NAME is not set in the environment variables.")

# Initialize MongoDB client
client = MongoClient(MONGODB_URL)
db = client[DATABASE_NAME]

# Access specific collections
chat_sessions = db["chat_sessions"]
chat_history = db["chat_history"]

# Function to get the database object
def get_db():
    return db
