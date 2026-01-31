import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

def clean_text(text):
    """
    Cleans garbled text by removing non-ASCII characters 
    and normalizing whitespace.
    """
    # 1. Remove non-ASCII characters (garbled symbols)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # 2. Replace multiple newlines or spaces with a single one
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def ingest_documents():
    # 1. Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # 2. List all available indexes to be sure
    active_indexes = pc.list_indexes().names()
    print(f"📡 Found these indexes in your account: {active_indexes}")
    
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    # Check if the index_name from .env actually exists
    if index_name not in active_indexes:
        if len(active_indexes) > 0:
            print(f"⚠️ Index '{index_name}' not found. Using '{active_indexes[0]}' instead.")
            index_name = active_indexes[0]
        else:
            print("❌ No indexes found at all. Creating a new one...")
            pc.create_index(
                name="auto-intel-index",
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            index_name = "auto-intel-index"


    # Path setup
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    raw_data_dir = BASE_DIR / "data"
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    for filename in os.listdir(raw_data_dir):
        if filename.endswith(".pdf"):
            file_path = raw_data_dir / filename
            print(f"🚀 Cleaning and Ingesting: {filename}")
            
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            
            # Apply cleaning to each page
            for page in pages:
                page.page_content = clean_text(page.page_content)
            
            # INCREASED CHUNK SIZE: 2000 characters helps keep tables together
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, 
                chunk_overlap=400
            )
            docs = text_splitter.split_documents(pages)
            
            PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
            print(f"✅ Successfully uploaded {len(docs)} clean chunks.")

if __name__ == "__main__":
    ingest_documents()