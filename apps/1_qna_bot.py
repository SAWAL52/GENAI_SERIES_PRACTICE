
from dotenv import load_dotenv
import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatOllama(
    model="llama3.2:1b",
    base_url="http://127.0.0.1:11434"
)

st.title("Hey Chatbot!! AI QnA Bot")
st.markdown("My QnA Bot with LangChain and Ollama")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

query = st.chat_input("Ask anything?")

if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    st.chat_message("user").markdown(query)

    chat_history = []

    for message in st.session_state.messages:
        if message["role"] == "user":
            chat_history.append(
                HumanMessage(content=message["content"])
            )
        elif message["role"] == "assistant":
            chat_history.append(
                AIMessage(content=message["content"])
            )

    response = llm.invoke(chat_history)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.content
    })

    st.chat_message("assistant").markdown(response.content)
