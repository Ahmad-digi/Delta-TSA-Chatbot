from langchain_chroma import Chroma
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from typing import List
import pandas as pd
import chromadb
from dotenv import load_dotenv
import sqlite3

# Load environment variables
load_dotenv()

# Clear system cache to avoid any potential issues
chromadb.api.client.SharedSystemClient.clear_system_cache()

def load_data() -> List[Document]:
    """
    Load data from a CSV file and return it as a list of Documents.
    """
    data = pd.read_csv("delta_electronics_ups.csv")
    return [
        Document(
            page_content=data.iloc[idx, :]["page_content"],
            metadata={
                "product_name": data.iloc[idx, :]["product_name"],
                "series_id": data.iloc[idx, :]["series_id"],
            },
        )
        for idx, row in data.iterrows()
    ]

def format_docs(docs: List[Document]) -> str:
    """
    Format the list of documents into a single string of context.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def create_chain():
    """
    Create and return a LangChain instance for generating responses.
    """
    openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Initialize the Chroma vector store
    vector_store = Chroma.from_documents(load_data(), embedding=openai_embeddings)
    retriever = vector_store.as_retriever()

    template = """You are a helpful, respectful, and honest Delta-TSA's Customer Support assistant. 
    Use the following pieces of context from the product documentation of Delta TSA's products to answer the question at the end. 
    You can only use the context to answer the user's question. 
    If there is no context or insufficient context to answer the user's question, apologize and prompt for further questions. 
    Do not hallucinate. If you know the answer, take a deep breath and explain your reasoning.
    If your answer can be enhanced using an image provided in the context, add it to the response as:
    ![Image Description](https://utltotheimage.png)

    <context>
    {context}
    </context>

    Question: {input}"""

    prompt = ChatPromptTemplate.from_template(template)

    # Configure the LLM (Language Model)
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
    )

    # Retrieve relevant documents for the input query
    retrieve_docs = (lambda x: x["input"]) | retriever

    # Define the RAG (Retriever-augmented Generation) chain
    rag_chain_from_docs = (
        {"input": lambda x: x["input"], "context": lambda x: format_docs(x["context"])}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Create the final chain
    chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=rag_chain_from_docs
    )
    return chain
