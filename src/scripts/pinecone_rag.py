from langchain_pinecone import PineconeVectorStore
from src.utils.embeddings import get_embeddings

def query_manual(query: str, index_name: str):
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding= get_embeddings()
    )

    docs = vectorstore.similarity_search(query, k=3)
    return docs