from pymongo import MongoClient
from sklearn.datasets import load_iris
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load iris dataset
iris = load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_data['target'] = iris.target

# Get MongoDB connection string from environment variables
connection_string = os.getenv('MONGO_URI')

# Connect to MongoDB
client = MongoClient(connection_string)

# Create database and collection
db = client['chatbot_database']
collection = db['iris_dataset']

# Convert DataFrame to records and upload
iris_records = iris_data.to_dict(orient='records')
collection.insert_many(iris_records)
