from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
import nemoguardrails as ngr
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()






OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")
# Initialize OpenRouter LLM
llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="meta-llama/llama-3.2-3b-instruct:free"
)

# completion = client.chat.completions.create(
#   model="openai/gpt-3.5-turbo",
#   messages=[
#     {
#       "role": "user",
#       "content": "What is the meaning of life?"
#     }
#   ]
# )
# print(completion.choices[0].message.content)

## Step 2: Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
index_name = "company-docs-index-test"
index_pc = None
if not pc.has_index(index_name):
    print('Creating index')
    index_pc = pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),)
    print('Index created')
else:
    index_pc = pc.Index(index_name)
    print('Index already exists')
    print("======================================")

## Step 3: Load and Process Company Documents
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

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50,length_function=len, is_separator_regex=False)
        splitted_texts = text_splitter.split_text(document.page_content)
        print(f"Page {i + 1} Data chunked into {len(splitted_texts)} chunks.")
        chunked_document.extend(splitted_texts)
    print(f"Total number of chunks: {len(chunked_document)}")
    print("=======================================")
    return chunked_document


pdf_path = r"./HR-Policies-Manuals.pdf"
document_pages = load_documents(file_path=pdf_path)
chunked_documents = chunk_document(loaded_document=document_pages)
print(f"Sample chunked document: {chunked_documents[0]}")
print("=======================================")

## Step 4: Generate Embeddings and Store in Vector Database
# embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))


# Step 5: Generate embeddings for each chunk

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

vectorstore = PineconeVectorStore.from_documents(documents_for_pinecone, index_name=index_name, embedding=hf)

# vectorstore = PineconeVectorStore.from_documents(chunked_documents, index_name=index_name,embedding=hf)

print(f"Stored embeddings in Pinecone index: {index_name}")
print("=======================================")

## Step 5: Define LLM and Retrieval Chain
# api_key_openai = os.getenv("OPENAI_API_KEY")
# if not api_key_openai:
#     raise ValueError("OPENAI_API_KEY environment variable not set")

# llm = OpenAI(api_key=api_key_openai, model="gpt-4")


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
# retrieval_qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     return_source_documents=True,
# )

## Step 6: Create a Prompt Template
# prompt_template = PromptTemplate(
#     input_variables=["question", "context"],
#     template="""
#     You are a helpful assistant specializing in answering questions based on the given context.
    
#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """
# )

# ## Step 7: Integrate Guardrails for Input and Output Validation
# rail_spec = """
# input {
#     query "User query must be relevant to company policies."
#     fail { "Response rejected: Please ensure your query relates to company policies." }
# }

# output {
#     answer "LLM output should be concise and relevant to the query."
# }
# """
# rail = ngr.Rails([rail_spec])

# def guarded_query(question):
#     response = rail.execute("query", {"query": question})
#     if not response["success"]:
#         return response["message"]
#     return question

## Step 8: Define Chatbot Functionality


system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
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


def chatbot_interface():
    print("Welcome to the Company Policy Chatbot!")
    while True:
        user_input = input("Ask your question: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        # validated_query = guarded_query(user_input)
        # if validated_query.startswith("Response rejected"):
        #     print(validated_query)
        #     continue
        # result = retrieval_qa.run(question=user_input)
        print(chain.invoke({"input": user_input}))
        # print("Answer:", result["answer"])
        # print("Source Documents:", result.get("source_documents", "Not Available"))

## Step 9: Run the Application
if __name__ == "__main__":
    chatbot_interface()

