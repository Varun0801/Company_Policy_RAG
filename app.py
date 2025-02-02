from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pinecone import Pinecone as pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# Initialize OpenRouter client with API Key and model settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="meta-llama/llama-3.2-3b-instruct:free"
)

# Step 2: Initialize Pinecone for vector storage
pc = pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
index_name = "company-docs-index-test"
index_pc = None

# Check if index exists
if not pc.has_index(index_name):
    print('Creating index...')
    index_pc = pc.create_index(
        name=index_name,
        dimension=768,  # Dimension of embeddings
        metric="cosine",  # Similarity metric
        spec={"cloud": "aws", "region": "us-east-1"},  # Cloud configuration
    )
    print('Index created.')

    # Step 3: Load and Process Company Documents
    def load_documents(file_path):
        """Load documents from a folder and preprocess them."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"Loaded the document.")
        print(f"Number of pages in document: {len(documents)}")
        print("=======================================")
        return documents

    def chunk_document(loaded_document):
        """Chunk the document into smaller sections."""
        chunked_document = []
        for i, document in enumerate(loaded_document):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50, length_function=len, is_separator_regex=False)
            splitted_texts = text_splitter.split_text(document.page_content)
            print(f"Page {i + 1} Data chunked into {len(splitted_texts)} chunks.")
            chunked_document.extend(splitted_texts)
        print(f"Total number of chunks: {len(chunked_document)}")
        print("=======================================")
        return chunked_document

    # Load PDF document
    pdf_path = r"./HR-Policies-Manuals.pdf"
    document_pages = load_documents(file_path=pdf_path)
    chunked_documents = chunk_document(loaded_document=document_pages)
    print(f"Sample chunked document: {chunked_documents[0]}")
    print("=======================================")

    # Step 4: Generate Embeddings and Store in Vector Database
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    chunk_embeddings = []
    for i, chunk in enumerate(chunked_documents):
        embedding = hf.embed_documents([chunk])
        chunk_embeddings.append({"id": f"chunk_{i}", "values": embedding[0]})
    print(f"Completed Generating embeddings for chunks.")
    print(f"Sample chunk embeddings: {chunk_embeddings[0]}")
    print("=======================================")

    # Step 6: Store embeddings in Pinecone
    documents_for_pinecone = [Document(page_content=chunk) for chunk in chunked_documents]
    vectorstore = Pinecone.from_documents(documents_for_pinecone, index_name=index_name, embedding=hf)

    print(f"Stored embeddings in Pinecone index: {index_name}")
    print("=======================================")

else:
    index_pc = pc.Index(index_name)
    print('Index already exists.')
    print("======================================")

# Step 5: Define the retrieval chain and setup LLM-based query handling
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Use Pinecone as a vector store directly
vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=hf)

# Setup retriever from the Pinecone vector store
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Create a prompt template for querying
system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

# Streamlit Interface for Continuous Conversation
import streamlit as st

st.title("Company Policy Chatbot")
st.write("Welcome to the Company Policy Chatbot! Ask any question related to company policies.")

# Initialize conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Display conversation history
for message in st.session_state.conversation_history:
    st.write(f"**{message['role']}**: {message['content']}")

# User input field for continuous conversation
user_input = st.text_input("Ask your question:")

if user_input:
    if user_input.lower() in ["bye", "exit"]:
        st.write("Goodbye!")
        st.session_state.conversation_history = []  # Clear conversation history
    else:
        # Add user message to conversation history
        st.session_state.conversation_history.append({"role": "User", "content": user_input})

        # Get response from LLM chain
        llm_response = chain.invoke({"input": user_input})

        # Add bot's response to the conversation history
        st.session_state.conversation_history.append({"role": "Bot", "content": llm_response['answer']})

        # Display bot's answer
        st.write(f"**Bot**: {llm_response['answer']}")
