import os
import uuid
from dotenv import load_dotenv
from src.agent.graph import app
from langchain_core.messages import HumanMessage

load_dotenv()

def run_autointel_agent():
    # LLMOps Checkpointing: Unique thread ID per session
    # This allows to resume conversations and track them in Langsmith
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("="*60)
    print("AutoIntel: Your Agentic Service Technician (2026)")
    print("Commands: 'quit' to exit, 'visualize' to see graph")
    print("="*60)

    while True:
        user_input = input("\n👤 User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye! Drive safely. 🚗")
            break
        
        if user_input.lower() == "visualize":
            try:
                print(app.get_graph().draw_ascii())
            except Exception as e:
                print(f"Could not visualize graph: {e}")
            continue

        # Start the agentic flow
        print("\n🤖 Processing...")

        # Use HumanMessage instead of tuple
        inputs = {"messages": [HumanMessage(content=user_input)]}

        try:
            # Invoke the agent with stream
            for event in app.stream(inputs, config, stream_mode="updates"):
                for node_name, output in event.items():
                    # LLMOps Logging: Show which node just fired
                    print(f"   [Node: {node_name}]")

                    # Check if output is not None and has messages
                    if output is not None and "messages" in output and len(output.get("messages", [])) > 0:
                        last_msg = output["messages"][-1]
                        
                        # Handle both tuple and Message object formats
                        if isinstance(last_msg, tuple):
                            content = last_msg[1]
                        elif hasattr(last_msg, 'content'):
                            content = last_msg.content
                        else:
                            content = str(last_msg)
                        
                        # Only print if it's from assistant (not echoing user input)
                        if node_name not in ["router", "safety_node"]:
                            print(f"\n🤖 Assistant: {content}\n")
                            
        except GeneratorExit:
            print("⚠️ Stream was interrupted")
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        run_autointel_agent()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye! 🚗")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()