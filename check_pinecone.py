# diagnostic_pinecone.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Initialize
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("auto-intel-index")

# Test queries
test_queries = [
    "What is the boot space capacity?",
    "What engine oil is recommended?",
    "tire pressure"
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)
    
    query_embedding = embeddings.embed_query(query)
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    
    for i, match in enumerate(results['matches'], 1):
        print(f"\n[Match {i}] Score: {match['score']:.3f}")
        print(f"ID: {match['id']}")
        print(f"Metadata keys: {list(match['metadata'].keys())}")
        
        # Print the actual text
        text = match['metadata'].get('text', match['metadata'].get('content', 'NO TEXT FOUND'))
        print(f"Text: {text[:300]}...")  # First 300 chars