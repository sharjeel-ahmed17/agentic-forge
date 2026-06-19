from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from config import api_key , base_url , model

memory = ConversationBufferMemory()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{history}\nUser: {input}")
])

llm = ChatOpenAI(model=model , base_url=base_url , api_key=api_key)

user_input = "My favorite language is Python"

history = memory.load_memory_variables({})["history"]

response = llm.invoke(
    prompt.format_messages(
        history=history,
        input=user_input
    )
)

memory.save_context(
    {"input": user_input},
    {"output": response.content}
)

print(response.content)