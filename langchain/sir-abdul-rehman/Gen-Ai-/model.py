from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

# API key load 
load_dotenv()

# Model setup 
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.8,
    max_tokens=20,
)

# sending the question to model through api
response = llm.invoke([HumanMessage(content="Write a peotry on Ai")])

# ans of llm in response
print(response.content)