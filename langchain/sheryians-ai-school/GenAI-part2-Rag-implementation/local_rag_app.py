# document load (image , text , pdf (text , text + image))
# chunking
# text spliting
# embedding model (multi model embedding -> cohere)
# vector store
# chat model
# prompt send to llm 
# conversation hist record 

from langchain_community.document_loaders import TextLoader , PyPDFLoader , UnstructuredImageLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from config import model , base_url , api_key


# ** load document text , image and pdf **

# for text document
text = TextLoader("notes.txt")
text_docs = text.load()

# for pdf text document 
pdf = PyPDFLoader("GRU.pdf")
pdf_docs = pdf.load()

# image load 
image = UnstructuredImageLoader(
    "example.png",
    mode="elements"
)
image_docs = image.load()
# chunking

splitter = RecursiveCharacterTextSplitter(
    separators="",
    chunk_size = 1000,
    chunk_overlap=1
)


# memory 
memory = ConversationBufferMemory()
# prompt
prompt = ChatPromptTemplate.from_messages([
    ("system" ,"you are helpful assistant."),
    ("human" , "{history}\nUser : {input}")
])

# chat model llm
model = init_chat_model(
    model_provider="openai",
    model=model,
    base_url=base_url,
    api_key=api_key
)


# embedding model
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
)


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