import os
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings # Change here
from langchain_core.tools import tool

@tool
def pinecone_rag_tool(query: str):
    """
    Consults the automobile user manuals to answer technical questions...
    """
    # Use HuggingFace instead of Groq for embeddings
    # This model is small, fast, and free to run locally or via API
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Connect to the existing Pinecone Index
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings
    )
    
    # Perform Similarity Search
    docs = vectorstore.similarity_search(query, k=3)
    
    # Format the results
    context = "\n\n".join([
        f"Source: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content}" 
        for d in docs
    ])
    
    return context