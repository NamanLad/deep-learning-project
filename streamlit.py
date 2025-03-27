import streamlit as st
# Set page config FIRST
st.set_page_config(page_title="MiniChatDev", page_icon="🤖", layout="wide")

# THEN import other modules
from chatdev_v9 import MiniChatDev, Groq
import json
import time

# Initialize the Groq client and MiniChatDev system
@st.cache_resource
def initialize_system():
    client = Groq(api_key="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")  # Replace "your-api-key" with actual key or use st.secrets
    return MiniChatDev(client)

# Initialize the system
chat_dev = initialize_system()

# Sidebar for settings
with st.sidebar:
    st.title("MiniChatDev Settings")
    max_iterations = st.slider("Max Improvement Iterations", 1, 5, 2)
    st.markdown("---")
    st.markdown("### Agents")
    st.markdown("- **CEO**: Defines requirements")
    st.markdown("- **Coder**: Writes and improves code")
    st.markdown("- **Tester**: Finds bugs")
    st.markdown("- **Reviewer**: Ensures code quality")
    st.markdown("---")
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
st.title("🤖 MiniChatDev - AI Software Development Team")
st.caption("Describe the software you want to build and watch the agents collaborate!")

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        if message["role"] == "coder" and "```python" in message["content"]:
            st.code(message["content"].split("```python")[1].split("```")[0], language="python")
        else:
            st.markdown(message["content"])

# User input
if prompt := st.chat_input("What software would you like to build?"):
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "avatar": "👤"
    })
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Start development process
    with st.spinner("Agents are collaborating..."):
        # Display agent responses as they come
        with st.chat_message("ceo", avatar="👔"):
            st.markdown("**CEO is defining requirements...**")
            ceo_placeholder = st.empty()
        
        # Run the project
        results, intermediate_results = chat_dev.run_project(prompt, max_iterations=max_iterations)
        
        # Update CEO message with actual content
        ceo_placeholder.markdown(f"**Requirements:**\n\n{results['requirements']}")
        
        # Add CEO message to history
        st.session_state.messages.append({
            "role": "ceo",
            "content": f"**Requirements:**\n\n{results['requirements']}",
            "avatar": "👔"
        })
        
        # Process intermediate results
        for i, result in enumerate(intermediate_results, start=1):
            for agent, message in result.items():
                print(agent, "\n")
                if agent == "requirements":
                    continue
                
                # Display each agent's message
                with st.chat_message(agent, avatar="👔" if agent == "ceo" else "👨‍💻" if agent == "code" else "🔍" if agent == "test_report" else "📝"):
                    if agent == "coder" and "```python" in message:
                        st.code(message.split("```python")[1].split("```")[0], language="python")
                    else:
                        st.markdown(message)
                
                # Add to message history
                st.session_state.messages.append({
                    "role": agent,
                    "content": message,
                    "avatar": "👔" if agent == "ceo" else "👨‍💻" if agent == "code" else "🔍" if agent == "test_report" else "📝"
                })

# Download button for results
if st.session_state.messages:
    st.download_button(
        label="Download Results",
        data=json.dumps(st.session_state.messages, indent=2),
        file_name="chatdev_conversation.json",
        mime="application/json"
    )