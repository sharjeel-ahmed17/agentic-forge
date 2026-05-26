import os
from dotenv import load_dotenv
load_dotenv()

def get_openai_config():
    return {
        "model": os.getenv("MODEL"),
        "base_url": os.getenv("BASE_URL"),
        "api_key": os.getenv("API_KEY")
    }


if __name__ == "__main__":
    config = get_openai_config()
    
