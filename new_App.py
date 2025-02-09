import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pinecone import Pinecone as pinecone
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Basic page config
st.set_page_config(
    page_title="Company Policy Assistant",
    page_icon="🤖"
)

# Initialize Streamlit state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'llm' not in st.session_state:
    st.session_state.llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="meta-llama/llama-3.2-3b-instruct:free"
    )

if 'qa_chain' not in st.session_state:
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    # Initialize Pinecone
    pc = pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV")
    )
    
    # Create vectorstore
    vectorstore = Pinecone.from_existing_index(
        index_name="company-docs-index-test",
        embedding=embeddings
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the given context to answer the question. If you don't know the answer, say you don't know. Use three sentences maximum and keep the answer concise. Context: {context}"),
        ("human", "{input}")
    ])
    
    # Create chain
    question_answer_chain = create_stuff_documents_chain(st.session_state.llm, prompt)
    st.session_state.qa_chain = create_retrieval_chain(retriever, question_answer_chain)

# App title
st.title("Company Policy Assistant")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question here:"):
    # Add user message to chat
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    try:
        # Get the response
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke({"input": prompt})
            
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response['answer'])
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()