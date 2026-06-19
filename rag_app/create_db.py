from langchain_community.document_loaders import TextLoader , PyPDFLoader , UnstructuredImageLoader , PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from config import cohere_api_key , cohere_model , VECTOR_DB_PATH


# ** load document text , image and pdf **

# for text document
text = TextLoader("notes.txt")
text_docs = text.load()

# for pdf text document 
pdf = PyPDFLoader("GRU.pdf")
pdf_docs = pdf.load()

# for pdf +  image document 
pdf_image = PyMuPDFLoader("GRU.pdf")
pdf_image_docs = pdf_image.load()

# image load 
image = UnstructuredImageLoader("example.png",mode="elements")
image_docs = image.load()


# debugging technique : 

for doc in text_docs:
    doc.metadata["source"] = "notes.txt"

for doc in pdf_docs:
    doc.metadata["source"] = "GRU.pdf"

for doc in image_docs:
    doc.metadata["source"] = "example.png"
    
# spliting anad chunking
all_docs = (text_docs , pdf_docs , pdf_image_docs , image_docs)
splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ],
    chunk_size = 500,
    chunk_overlap=50
)
chunks = splitter.split_documents(all_docs)


# embedding model
embedding_model = CohereEmbeddings(
    model=cohere_model,
    cohere_api_key=cohere_api_key
)

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model

)

# save vector db 
vectorstore.save_local(VECTOR_DB_PATH)