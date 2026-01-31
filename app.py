import streamlit as st
from langchain_core.messages import HumanMessage
from src.agent.graph import app
import uuid

st.set_page_config(page_title="AutoIntel AI Assistant", page_icon="🚗")

st.title("🚗 AutoIntel: Your AI Service Technician")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your vehicle..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Prepare config
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Show thinking indicator
            with st.spinner("Thinking..."):
                # Stream the response
                full_response = ""
                
                for event in app.stream(
                    {"messages": [HumanMessage(content=prompt)]},
                    config,
                    stream_mode="updates"
                ):
                    for node_name, output in event.items():
                        # ✅ FIX: Check if output is not None and has messages
                        if output is not None and isinstance(output, dict) and "messages" in output:
                            messages = output.get("messages", [])
                            if messages and len(messages) > 0:
                                last_msg = messages[-1]
                                
                                # Extract content
                                if isinstance(last_msg, tuple):
                                    content = last_msg[1]
                                elif hasattr(last_msg, 'content'):
                                    content = last_msg.content
                                else:
                                    content = str(last_msg)
                                
                                # Only show final response (not intermediate nodes)
                                if node_name not in ["router"]:
                                    full_response = content
                                    message_placeholder.markdown(content)
                
                # If no response was generated, show a default message
                if not full_response:
                    full_response = "I apologize, but I couldn't generate a response. Please try again."
                    message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            st.error(error_msg)
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This AI assistant can help you with:
    - 📖 Service manual information
    - 🔧 Maintenance guidelines
    - 🚨 Recall information (via VIN)
    - ❓ General vehicle questions
    """)
    
    st.divider()
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    
    st.divider()
    
    st.markdown("**Thread ID:**")
    st.code(st.session_state.thread_id, language=None)