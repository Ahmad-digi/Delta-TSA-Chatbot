import streamlit as st
import sqlite3
from chain import create_chain  # Import your chain creation logic
import chromadb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Clear system cache to avoid any potential issues
chromadb.api.client.SharedSystemClient.clear_system_cache()

# App title and configuration
st.set_page_config(page_title="Delta TSA Chatbot", page_icon="🤖")

# Create the LangChain instance once
chain = create_chain()

# Sidebar and instructions
with st.sidebar:
    st.title("Delta TSA Chatbot")
    st.markdown(
        "This chatbot uses a custom LangChain-powered backend for answering queries."
    )
    if st.button("Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function to generate a response
def generate_response(prompt):
    """
    Use the custom LangChain implementation to generate a response.
    """
    try:
        response = chain.invoke({"input": prompt})["answer"]
        return response
    except Exception as e:
        return f"An error occurred: {e}"

# User input
if prompt := st.chat_input("Type your message here..."):
    # Add user input to the session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate the assistant's response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt)
                st.write(response)
        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
