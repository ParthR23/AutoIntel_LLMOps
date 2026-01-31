# src/agent/nodes.py
from src.tools.car_api import car_service_api
from src.agent.state import VehicleDetails, AgentState
from langchain_groq import ChatGroq
from src.tools.pinecone_rag import pinecone_rag_tool
from src.tools.car_review import car_review_tool
from langchain_core.messages import AIMessage

# Initialize LLM once at module level
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
extractor = llm.with_structured_output(VehicleDetails)

# RAG System Prompt
RAG_SYSTEM_PROMPT = """
You are a highly skilled Hyundai Service Assistant. 
Your goal is to provide accurate technical information based ONLY on the provided manual context.

CONTEXT FROM MANUAL:
{context}

USER QUESTION: 
{question}

INSTRUCTIONS:
1. Use the context above to answer the user's question accurately.
2. If the answer is not found in the context, strictly say: "I'm sorry, I don't have information about that in the service manual."
3. If the context contains specific numbers (like "256 Litres" for boot space), ensure you mention them.
4. Keep your tone professional and helpful.
"""

def call_rag(state):
    """
    RAG node that retrieves from Pinecone and generates answer.
    """
    messages = state.get("messages", [])
    
    if messages:
        last_msg = messages[-1].content
    else:
        last_msg = "No question found."
    
    print(f"🔍 Searching manuals for: {last_msg}")
    
    try:
        context = pinecone_rag_tool.invoke(last_msg)
        print(f"📄 Retrieved context (first 200 chars): {context[:200]}...")
        
        formatted_prompt = RAG_SYSTEM_PROMPT.format(context=context, question=last_msg)
        response = llm.invoke(formatted_prompt)
        
        print(f"💬 LLM Response: {response.content[:200]}...")
        
        if isinstance(response, str):
            return {"messages": [AIMessage(content=response)]}
        else:
            return {"messages": [response]}
            
    except Exception as e:
        print(f"Error in call_rag: {e}")
        return {"messages": [AIMessage(content="I'm sorry, I encountered an error while searching the manual. Please try again.")]}

def call_api(state):
    """
    Uses an LLM to extract vehicle details, then calls the NHTSA API.
    """
    messages = state.get("messages", [])
    
    if not messages:
        return {"messages": [AIMessage(content="No message found.")]}
    
    if hasattr(messages[-1], 'content'):
        last_message = messages[-1].content
    else:
        last_message = str(messages[-1])
    
    print(f"🚗 Extracting vehicle details from: {last_message}")
    
    extraction_prompt = f"""Extract the vehicle information from this question.
    
Question: {last_message}

Instructions:
- Extract the MAKE (brand): e.g., BMW, Toyota, Honda
- Extract the MODEL: e.g., 3 Series, Camry, Civic (if mentioned)
- Extract the YEAR: e.g., 2024, 2023 (REQUIRED)

If the model is not mentioned, use the make name.
Return the information in a structured format."""
    
    try:
        vehicle_info = extractor.invoke(extraction_prompt)
        print(f"✅ Extracted: Year={vehicle_info.year}, Make={vehicle_info.make}, Model={vehicle_info.model}")

        if not vehicle_info.year:
            return {"messages": [AIMessage(content="I need the vehicle year to check for recalls. Please specify the year (e.g., '2024 BMW recalls').")]}
        
        if not vehicle_info.make:
            return {"messages": [AIMessage(content="I need the vehicle make/brand to check for recalls. Please specify the manufacturer.")]}
        
        if not vehicle_info.model or vehicle_info.model.lower() in ["unknown", "not specified"]:
            vehicle_info.model = vehicle_info.make
            print(f"⚠️ Model not specified, using make as model: {vehicle_info.model}")

        api_response = car_service_api.invoke({
            "make": vehicle_info.make,
            "model": vehicle_info.model,
            "year": vehicle_info.year
        })
        
        return {"messages": [AIMessage(content=str(api_response))]}

    except Exception as e:
        print(f"❌ Error in call_api: {e}")
        return {"messages": [AIMessage(content="I couldn't extract the vehicle information. Please provide the year, make, and model.")]}

def call_review(state):
    """
    Fetches car reviews and comparisons from Car and Driver.
    """
    messages = state.get("messages", [])
    
    if not messages or len(messages) == 0:
        return {"messages": [AIMessage(content="No message found.")]}
    
    if hasattr(messages[-1], 'content'):
        last_message = messages[-1].content
    else:
        last_message = str(messages[-1])
    
    print(f"📰 Fetching car review for: {last_message}")
    
    try:
        review_response = car_review_tool.invoke(last_message)
        return {"messages": [AIMessage(content=review_response)]}
        
    except Exception as e:
        print(f"❌ Error in call_review: {e}")
        import traceback
        traceback.print_exc()
        return {"messages": [AIMessage(content=f"I encountered an error while fetching reviews. Please try searching on caranddriver.com directly.")]}