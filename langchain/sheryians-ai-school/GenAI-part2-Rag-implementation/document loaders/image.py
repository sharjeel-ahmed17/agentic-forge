from langchain_community.document_loaders import UnstructuredImageLoader
import os

image_path = "example.png"

# Check file exists
print("File exists:", os.path.exists(image_path))
print("File size:", os.path.getsize(image_path), "bytes")

loader = UnstructuredImageLoader(
    image_path,
    mode="elements",
    language="eng"
)

docs = loader.load()

print("Total docs:", len(docs))

if not docs:
    print("⚠️  No content extracted. Check Tesseract install & image quality.")

for i, doc in enumerate(docs):
    print(f"\n--- DOC {i} ---")
    print(doc.page_content)
    print(doc.metadata)