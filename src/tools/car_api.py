import os
import requests
from typing import Union, List, Optional
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- RAG TOOL SETUP ---

def get_pinecone_retriever():
    """Initializes and returns the Pinecone vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings
    )
    return vectorstore

@tool
def pinecone_rag_tool(query: str) -> str:
    """
    Search the car's technical service manual for specific data.
    Use this for technical specs, maintenance schedules, or interior features.
    """
    try:
        vectorstore = get_pinecone_retriever()
        docs = vectorstore.similarity_search(query, k=4)
        context = "\n---\n".join([doc.page_content for doc in docs])
        return context if context else "No relevant information found in the manual."
    except Exception as e:
        return f"Error accessing manual: {str(e)}"

# --- REVIEW TOOL SETUP ---

@tool
def car_review_search(query: str):
    """
    Search for car reviews and comparisons on Car and Driver.
    Use this for expert opinions, performance specs, and competitor comparisons.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY is missing. Please add it to your .env file."

    try:
        # Initializing inside the tool to ensure it catches environment changes
        search = TavilySearchResults(
            tavily_api_key=api_key,
            max_results=3,
            search_depth="advanced",
            include_domains=["caranddriver.com"]
        )
        return search.invoke({"query": f"{query} site:caranddriver.com"})
    except Exception as e:
        return f"Car and Driver search failed: {str(e)}"

# --- RECALL API TOOL SETUP ---

class CarServiceInput(BaseModel):
    """Input schema for the car service API."""
    model_config = ConfigDict(coerce_numbers_to_str=True)
    make: str = Field(description="Car manufacturer, e.g., 'BMW'")
    model: str = Field(description="Specific model name, e.g., '3 Series'")
    year: Union[str, int] = Field(description="Manufacturing year")

@tool(args_schema=CarServiceInput)
def car_service_api(make: str, model: str, year: Union[str, int]) -> str:
    """
    Queries the official NHTSA database for safety recalls.
    """
    year_str = str(year)
    make_up = make.strip().upper()
    model_up = model.strip().upper()

    def call_nhtsa(mk: str, md: str, yr: str):
        url = f"https://api.nhtsa.gov/recalls/recallsByVehicle?make={mk}&model={md}&modelYear={yr}"
        try:
            response = requests.get(url, timeout=10)
            return response.json() if response.status_code == 200 else None
        except:
            return None

    # Try 1: Exact Match
    data = call_nhtsa(make_up, model_up, year_str)

    # Try 2: Smart Fallback for specific models
    if not data or data.get('Count') == 0:
        if any(x in model_up for x in ["330", "340", "M3"]):
            data = call_nhtsa(make_up, "3 SERIES", year_str)
        elif "GRAND I10" in model_up:
            data = call_nhtsa(make_up, "I10", year_str)

    if not data or data.get('Count') == 0:
        return f"No safety recalls found in the NHTSA database for the {year_str} {make_up} {model_up}."

    # Format Results
    count = data['Count']
    summary = f"⚠️ Found {count} recall(s) for the {year_str} {make_up} {model_up}:\n"
    for i, r in enumerate(data['results'][:3], 1):
        summary += f"\n{i}. {r.get('Component')}: {r.get('Summary')[:200]}..."
    
    return summary