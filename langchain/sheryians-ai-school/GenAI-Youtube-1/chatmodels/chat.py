from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from rich import print

model = ChatMistralAI(model = "mistral-small-2506",temperature=0.9)

response = model.invoke("write a poem on AI")

print(response.content)