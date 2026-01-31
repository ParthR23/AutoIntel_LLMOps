from typing import TypedDict, Annotated, Sequence, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class VehicleDetails(BaseModel):
    """Vehicle information extracted from user query."""
    year: Optional[int] = Field(
        default=None,
        description="The vehicle year (e.g., 2024, 2023). REQUIRED for recall checks."
    )
    make: Optional[str] = Field(
        default=None,
        description="The vehicle manufacturer/brand (e.g., BMW, Toyota, Honda, Hyundai). REQUIRED."
    )
    model: Optional[str] = Field(
        default=None,
        description="The vehicle model (e.g., '3 Series', 'Camry', 'Creta'). Use the make if model is not specified."
    )

class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_action: str