""" Simple RAG Pipeline using NVIDIA NIM, NVIDIA langchain connectors and FAISS vector store"""

import os

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever

## Setup
nvapi_key = ""

## Load Data
pdf_directory = "pdf-data/"
pdf_docs = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
data_docs = []

for pdf in pdf_docs:
    loader = UnstructuredPDFLoader(file_path=pdf)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    docs = loader.load_and_split(text_splitter=splitter)
    data_docs +=  docs

## Define pipeline components: LLM, Embedder, Reranker
llm = ChatNVIDIA(model="meta/llama2-70b", nvidia_api_key=nvapi_key)

embedder = NVIDIAEmbeddings(
  model="nvidia/nv-embedqa-e5-v5", 
  api_key=nvapi_key, 
  truncate="NONE", 
)

reranker = NVIDIARerank(
  model="nv-rerank-qa-mistral-4b:1", 
  api_key=nvapi_key,
)

## create vector store and add docs
db = FAISS.from_documents(data_docs, embedder)

## Define compression retriever for retrieval + reranking
retriever = db.as_retriever()

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=retriever
)

## define chain
chain = llm | StrOutputParser()

def ask_question(question):
    """returns a response from the llm"""
    docs = compression_retriever.invoke(question)
    context = ""
    for doc in docs:
        context += doc.page_content + "\n\n"
    
    ans = chain.invoke("context: " + context + "question: " + question)
    return ans

q = "What topics will be on the exam?"
print(ask_question(q))

