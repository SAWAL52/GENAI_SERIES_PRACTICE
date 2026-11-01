
import streamlit as st
from dotenv import load_dotenv

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import Tool

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="Google Search AI Agent",
    page_icon="🔎",
    layout="centered"
)

st.title("🔎 Google Search AI Agent")
st.caption("Ask questions and get answers using Google Search and Groq.")

# Initialize agent only once
@st.cache_resource
def get_agent():

    model = ChatGroq(
        model="openai/gpt-oss-20b",
        streaming=True
    )

    search = GoogleSerperAPIWrapper()

    search_tool = Tool(
        name="google_search",
        func=search.run,
        description="Search Google for current information and answer user questions."
    )

    agent = create_agent(
        model=model,
        tools=[search_tool],
        system_prompt=(
            "You are a helpful research assistant. "
            "Use Google Search whenever external or current information is needed. "
            "Give clear and concise answers based on search results."
        )
    )

    return agent


agent = get_agent()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
query = st.chat_input("Ask anything...")

if query:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching Google and generating answer..."):

            try:
                response = agent.invoke({
                    "messages": [
                        {"role": "user", "content": query}
                    ]
                })

                answer = response["messages"][-1].content

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write(
        "This chatbot uses LangChain, Groq and Google Serper "
        "to search the web and answer questions."
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
