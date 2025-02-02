from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="",
)

completion = client.chat.completions.create(
  model="meta-llama/llama-3.2-3b-instruct:free",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)
print(completion.choices[0].message.content)

# import os
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain_openai import OpenAI
# from langchain_community.vectorstores import Pinecone
# from langchain_community.embeddings import OpenAIEmbeddings
# from pinecone import Pinecone, ServerlessSpec

# # Load API keys
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# # Initialize Pinecone client
# pc = Pinecone(api_key=PINECONE_API_KEY)

# # Ensure index exists
# index_name = "company-policy-index"
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=1536,  # Ensure this matches the embedding model
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-west-2")
#     )

# # Connect to the existing Pinecone index
# vectorstore = Pinecone.from_existing_index(index_name, OpenAIEmbeddings())

# # Load stored embeddings (Ensures embeddings are created only once)
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# # Initialize OpenRouter LLM
# llm = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=OPENROUTER_API_KEY,
#     model="meta-llama/llama-3.2-3b-instruct:free"
# )

# # Define the prompt template
# prompt_template = PromptTemplate(
#     input_variables=["question", "context"],
#     template="""
#     You are an AI assistant that answers questions based on company policy documents.

#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """
# )

# # Create LLM Chain
# llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# # Function to process queries
# def query_llm(user_input):
#     retrieved_docs = retriever.invoke(user_input)  # Updated for LangChain v1.0+
#     context = "\n\n".join([doc.page_content for doc in retrieved_docs])

#     # Ensure OpenRouter receives the correct `prompt`
#     result = llm_chain.invoke({"question": user_input, "context": context})
#     return result["text"]  # Extract response text

# # Chatbot interface
# def chatbot_interface():
#     print("Welcome to the Company Policy Chatbot!")
#     while True:
#         user_input = input("Ask your question: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("Goodbye!")
#             break
#         answer = query_llm(user_input)
#         print("Answer:", answer)

# # Run chatbot
# if __name__ == "__main__":
#     chatbot_interface()

