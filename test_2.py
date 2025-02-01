import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

index_name = "company-docs-index"
embeddings = OpenAIEmbeddings()

# path to an example pdf file
loader = PyPDFLoader("HR-Policies-Manuals.pdf")
documents = loader.load()
print(f"Loaded the document.")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
print(f"Total number of chunks: {len(docs)}")

vectorstore_from_docs = PineconeVectorStore.from_documents(
    docs,
    index_name=index_name,
    embedding=embeddings
)
print("Vector store created")

# vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

query = "Internship Programme"
result = vectorstore_from_docs.similarity_search(query)
print(result)
