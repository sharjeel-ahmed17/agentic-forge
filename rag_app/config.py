from dotenv import load_dotenv
load_dotenv()
import os
base_url = os.getenv("BASE_URL")
model  = os.getenv("MODEL")
api_key = os.getenv("API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_model = os.getenv("COHERE_MODEL")
VECTOR_DB_PATH = "vector_db"

if not(base_url and model and api_key):
    raise ValueError("provide baseurl  , model and api key ")

