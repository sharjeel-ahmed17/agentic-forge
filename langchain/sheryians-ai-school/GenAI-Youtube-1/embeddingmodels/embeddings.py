from dotenv import load_dotenv
load_dotenv()
# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import json

# embeddings = OpenAIEmbeddings(
#     model = 'text-embedding-3-large',
#     dimensions=64
# )
embeddings = GoogleGenerativeAIEmbeddings(
    model = 'gemini-embedding-2-preview',
    output_dimensions=768
)

texts = [
    "Hello this is Akarsh Vyas",
    "Hello your name is YouTube",
    "And you all are very beautiful"
]

vector = embeddings.embed_documents(texts)

print(vector)

with open('embeddings.json','w') as file:
    json.dump(vector,file )
