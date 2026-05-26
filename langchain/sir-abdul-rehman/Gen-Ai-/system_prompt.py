from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from dotenv import load_dotenv
import os

# API key load 
load_dotenv()

# Model setup 
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.8,
)

# System prompt 
# prompt=input("You: ")
# response = llm.invoke(prompt)

# # ans of llm in response
# print("Bot: ",response.content)


# system prompt 
print("------------------ Ai Bot ------------------")
print("       Type 'exit' to quit the chat         ")

# function to make a loop of the chat , until user input is exit
while True:
    prompt=input("You: ")
    if prompt=='exit':
        break
    response = llm.invoke(prompt)
    print("Bot: ",response.content) 
