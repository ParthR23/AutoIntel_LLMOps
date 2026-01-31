import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
load_dotenv()


# Specialized Safety Model
safety_model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def is_content_safe(content: str) -> bool:
    """
    Checks if the content violates safety policies using Llama Guard.
    """
    # Llama Guard expects a specific prompt format
    # It checks for: Violence, Sexual Content, Criminal Advice, etc.
    response = safety_model.invoke([HumanMessage(content=content)])
    
    # Llama Guard returns 'safe' or 'unsafe\n<category>'
    decision = response.content.strip().lower()
    
    if "unsafe" in decision:
        return False
    return True