from langchain_community.document_loaders import UnstructuredImageLoader
loader = UnstructuredImageLoader(
    "image.png",
    mode="elements"
)

docs = loader.load()

for doc in docs:
    print(doc.metadata.get("category"))
    print(doc.page_content)