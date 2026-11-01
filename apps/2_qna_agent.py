
import streamlit as st
import time
import uuid

from dotenv import load_dotenv

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import MemorySaver


st.set_page_config(
    page_title="Google Search AI Agent",
    page_icon="🔎",
    layout="centered"
)

load_dotenv()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("🔎 Google Search AI Agent")
st.caption("Ask questions and get answers using Google Search and Groq.")


@st.cache_resource
def get_agent():

    model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        streaming=True
    )

    search = GoogleSerperAPIWrapper()

    search_tool = Tool(
        name="google_search",
        func=search.run,
        description=(
            "Search Google for current information. "
            "Use this tool when external or current information is needed."
        )
    )

    agent = create_agent(
        model=model,
        tools=[search_tool],
        system_prompt=(
            "You are a helpful research assistant. "
            "Use Google Search whenever external or current "
            "information is needed. "
            "Give clear and concise answers based on search results."
        ),
        checkpointer=MemorySaver()
    )

    return agent


agent = get_agent()


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


query = st.chat_input("Ask anything...")


if query:

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        placeholder = st.empty()
        start_time = time.perf_counter()

        message = ""

        try:

            response = agent.stream(
                {
                    "messages": [
                        {"role": "user", "content": query}
                    ]
                },
                {
                    "configurable": {
                        "thread_id": st.session_state.thread_id
                    }
                },
                stream_mode="messages"
            )

            for chunk, metadata in response:

                if metadata.get("langgraph_node") != "model":
                    continue

                content = chunk.content

                if isinstance(content, str):
                    text_chunk = content

                elif isinstance(content, list):
                    text_chunk = "".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict)
                        and block.get("type") == "text"
                    )

                else:
                    text_chunk = ""

                if text_chunk:
                    message += text_chunk
                    placeholder.markdown(message + "▌")

            end_time = time.perf_counter()

            response_time = end_time - start_time

            placeholder.markdown(message)

            st.caption(
                f"⏱️ Response time: {response_time:.2f} seconds"
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": message
            })

        except Exception as e:
            st.error(f"Something went wrong: {e}")


with st.sidebar:

    st.header("About")

    st.write(
        "This chatbot uses LangChain, Groq and Google Serper "
        "to search the web and answer questions."
    )

    if st.button("Clear Chat"):

        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())

        st.rerun()
