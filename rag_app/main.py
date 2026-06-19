# document load (image , text , pdf (text , text + image))
# chunking
# text spliting
# embedding model (multi model embedding -> cohere)
# vector store
# chat model
# prompt send to llm 
# conversation hist record 

from langchain_community.document_loaders import TextLoader , PyPDFLoader , UnstructuredImageLoader , PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from config import model , base_url , api_key , cohere_api_key , cohere_model , VECTOR_DB_PATH

# memory 
memory = ConversationBufferMemory()

# ** load document text , image and pdf **

# for text document
text = TextLoader("notes.txt")
text_docs = text.load()

# for pdf text document 
pdf = PyPDFLoader("GRU.pdf")
pdf_docs = pdf.load()

# for pdf +  image document 
pdf_image = PyMuPDFLoader("GRU.pdf")
pdf_image_docs = pdf.load()

# image load 
image = UnstructuredImageLoader("example.png",mode="elements")
image_docs = image.load()

# spliting anad chunking
list_of_docs = [text_docs , pdf_docs , pdf_image_docs , image_docs]
splitter = RecursiveCharacterTextSplitter(
    separators="",
    chunk_size = 1000,
    chunk_overlap=1
)
chunks = splitter.split_documents(list_of_docs)


# embedding model
embedding_model = CohereEmbeddings(
    model=cohere_model,
    cohere_api_key=cohere_api_key
)

vectorstore = FAISS(
    embedding_function=embedding_model

)

# save vector db 
vectorstore.save_local(VECTOR_DB_PATH)
retriever = vectorstore.as_retriever(
    search_type = "mmr",
        search_kwargs = {
            "k" : 4,
            "fetch_k":10,
            "lambda_mult" :0.5
        }
)
# chat model llm
model = init_chat_model(
    model_provider="openai",
    model=model,
    base_url=base_url,
    api_key=api_key
)




# prompt
prompt = ChatPromptTemplate.from_messages([
    ("system" ,"you are helpful assistant."),
    ("human" , "{history}\nUser : {input}")
])

# ! memory referce start
# user_input = "My favorite language is Python"
# history = memory.load_memory_variables({})["history"]

# response = model.invoke(
#     prompt.format_messages(
#         history=history,
#         input=user_input
#     )
# )

# memory.save_context(
#     {"input": user_input},
#     {"output": response.content}
# )

# print(response.content)

# ! memory referce end