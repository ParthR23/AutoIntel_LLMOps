import os 
from typing import TypedDict, Annotated, Sequence
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START   
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from src.agent.safety import is_content_safe
from src.agent.nodes import call_rag, call_api, call_review

# Initialize the LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_action: str

# Router Node
def router_node(state: AgentState):
    """
    Analyses the user query and decides: 'rag', 'api', or 'review'.
    """
    messages = state.get("messages", [])
    if not messages or len(messages) == 0:
        return {"next_action": "rag"}
    
    if hasattr(messages[-1], 'content'):
        msg = messages[-1].content.lower()
    else:
        msg = str(messages[-1]).lower()

    print(f"🔀 Router analyzing: {msg[:100]}...")
    
    # Enhanced review keywords
    review_keywords = [
        "review", "reviews", 
        "comparison", "compare", "vs", "versus",
        "better", "best", "top",
        "rating", "ratings",
        "opinion", "thoughts",
        "worth it", "worth buying",
        "should i buy", "should i get",
        "recommend", "recommendation",
        "alternatives", "options",
        "which car", "which suv", "which sedan",
        "luxury", "affordable", "budget",
        "reliable", "most reliable"
    ]
    
    if any(keyword in msg for keyword in review_keywords):
        print(f"   → Routing to REVIEW")
        return {"next_action": "review"}
    
    # API keywords
    api_keywords = [
        "recall", "recalls", "recalled",
        "vin", 
        "service history", 
        "mileage",
        "safety issue", "safety issues",
        "defect", "defects",
        "nhtsa"
    ]
    
    if any(keyword in msg for keyword in api_keywords):
        print(f"   → Routing to API")
        return {"next_action": "api"}
    
    # Default to RAG
    print(f"   → Routing to RAG")
    return {"next_action": "rag"}

# Safety Check Node
def safety_check_node(state: AgentState):
    """
    The final 'Border Control' for all assistant messages.
    """
    messages = state.get("messages", [])
    if not messages or len(messages) == 0:
        return {}
    
    # Get last message content safely
    if hasattr(messages[-1], 'content'):
        last_message = messages[-1].content
    else:
        last_message = str(messages[-1])
    
    # Check safety
    try:
        if not is_content_safe(last_message):
            redacted_msg = AIMessage(content="⚠️ I'm sorry, but I cannot provide that information as it violates my safety policy regarding vehicle security or dangerous procedures.")
            return {"messages": [redacted_msg]}
    except Exception as e:
        print(f"⚠️ Safety check failed: {e}")
        # If safety check fails, allow the message through
        pass
    
    # If safe, return empty dict (no changes)
    return {}

# Build the Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("router", router_node)
workflow.add_node("rag_node", call_rag)
workflow.add_node("api_node", call_api)
workflow.add_node("review_node", call_review)
workflow.add_node("safety_node", safety_check_node)

# Define edges
workflow.add_edge(START, "router")

# Conditional routing from router
workflow.add_conditional_edges(
    "router", 
    lambda x: x.get("next_action", "rag"), 
    {
        "rag": "rag_node", 
        "api": "api_node",
        "review": "review_node"
    }
)

# All tools flow to Safety before ending
workflow.add_edge("rag_node", "safety_node")
workflow.add_edge("api_node", "safety_node")
workflow.add_edge("review_node", "safety_node")
workflow.add_edge("safety_node", END)

# Compile with Persistence
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)