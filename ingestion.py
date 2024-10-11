from dotenv import load_dotenv
import os
import sys
import io

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from consts import INDEX_NAME

load_dotenv()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs():
    #loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest", encoding='utf-8')
    loader = ReadTheDocsLoader("rubik-docs/kubekings.com/cubos-de-rubik-3x3", patterns="*.html", custom_html_tag=('html', {}), encoding='utf-8')
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    print(f"split {len(documents)} documents")
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("rubik-docs", "https://")
        new_url = new_url.replace("\\", "/")
        doc.metadata.update({"source": new_url})

    print(f"going to add {len(documents)} to pinecone")

    PineconeVectorStore.from_documents(
        documents, embeddings, index_name=INDEX_NAME
    )

if __name__ == "__main__":
    ingest_docs()