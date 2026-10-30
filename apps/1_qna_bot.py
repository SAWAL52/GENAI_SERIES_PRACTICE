
from dotenv import load_dotenv
import streamlit as st

from langchain_ollama import ChatOllama

load_dotenv()

# Initialize Ollama
llm = ChatOllama(
    model="llama3.2:1b",
    base_url="http://127.0.0.1:11434"
)

# Streamlit UI
st.title("Hey Chatbot!! AI QnA Bot")
st.markdown("My QnA Bot with LangChain and Ollama")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# User input
query = st.chat_input("Ask anything?")

if query:
    # Display and save user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    st.chat_message("user").markdown(query)

    # Get AI response
    response = llm.invoke(query)

    # Display and save AI response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.content
    })

    st.chat_message("assistant").markdown(response.content)
