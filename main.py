# def main():
#     print("Hello from generative-ai!")


# if __name__ == "__main__":
#     main()
from openai_provider import get_openai_config

provider = get_openai_config()

provider["model"]
provider["base_url"]
provider["api_key"]
