from dotenv import load_dotenv
load_dotenv()
import os
base_url = os.getenv("BASE_URL")
model  = os.getenv("MODEL")
api_key = os.getenv("API_KEY")

if not(base_url and model and api_key):
    raise ValueError("provide baseurl  , model and api key ")

