from typing import TypedDict, Annotated, List
from Langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The chat history"]
    category: str
    car_model: str
    response: str