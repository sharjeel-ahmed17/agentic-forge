# document load (image , text , pdf (text , text + image))
# chunking
# text spliting
# embedding model (multi model embedding -> cohere)
# vector store
# chat model
# prompt send to llm 
# conversation hist record 


from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from config import model , base_url , api_key , cohere_api_key , cohere_model , VECTOR_DB_PATH

# memory 
memory = ConversationBufferMemory()

# embedding model
embedding_model = CohereEmbeddings(
    model=cohere_model,
    cohere_api_key=cohere_api_key
)
vectorstore = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)


retriever = vectorstore.as_retriever(
    search_type = "mmr",
        search_kwargs = {
            "k" : 4,
            "fetch_k":10,
            "lambda_mult" :0.5
        }
)
# chat model llm
model = init_chat_model(model_provider="openai",model=model,base_url=base_url,api_key=api_key)

# prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.
            Use ONLY the provided context to answer the question.
            If the answer is not present in the context,
            say: "I could not find the answer in the document.
            """
        ),
        (
            "human",

            """
            Conversation History:{history}
            
            Context: {context}

            Question: {question}

            """
        )
    ]
)


print("Rag system created ")

print("press 0 to exit ")

while True:
    query = input("You : ")
    history = memory.load_memory_variables({})["history"]
    if query == "0":
        break 
    
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query,
        "history" : history,
    })
    
    response = model.invoke(final_prompt)


    print(f"\n AI: {response.content}")

# save memory conversation
    memory.save_context(
    {"input": query},
    {"output": response.content}
)


