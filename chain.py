from langchain_chroma import Chroma
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from typing import List
import pandas as pd
from dotenv import load_dotenv
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

load_dotenv()

def load_data() -> List[Document]:
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
    return "\n\n".join(doc.page_content for doc in docs)


def create_chain():
    openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


    vector_store = Chroma.from_documents(load_data(), embedding=openai_embeddings)
    retriever = vector_store.as_retriever()

    template = """You are a helpful, respectful and honest Delta-TSA's Customer Support assistant. 
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

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
    )

    retrieve_docs = (lambda x: x["input"]) | retriever

    rag_chain_from_docs = (
        {"input": lambda x: x["input"], "context": lambda x: format_docs(x["context"])}
        | prompt
        | llm
        | StrOutputParser()
    )

    chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=rag_chain_from_docs
    )
    return chain
