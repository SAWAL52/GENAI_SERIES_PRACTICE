
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent


st.set_page_config(
    page_title="Task Management AI",
    page_icon="📝",
    layout="centered"
)


if "thread_id" not in st.session_state:
    st.session_state.thread_id = "1"

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def get_agent():

    db = SQLDatabase.from_uri("sqlite:///my_tasks.db")

    db.run("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT CHECK(status IN ('pending', 'completed', 'in_progress')) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        streaming=True
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    tools = toolkit.get_tools()

    memory = InMemorySaver()

    system_prompt = """
    You are a task management assistant that interacts with a SQL Database containing a 'tasks' table.

    TASK RULES:
    1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC.
    2. After CREATE, UPDATE, or DELETE, confirm with a SELECT query.
    3. If the user requests a list of tasks, present the output in a structured table format.
    4. After every reply, present the table with current tasks subject to rule no. 1.

    CRUD OPERATIONS:
    CREATE: INSERT INTO tasks(title, description, status)
    READ: SELECT * FROM tasks WHERE ... LIMIT 10
    UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
    DELETE: DELETE FROM tasks WHERE id=? OR title=?

    TABLE SCHEMA:
    id, title, description, status (pending/in_progress/completed), created_at.
    """

    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=memory,
        system_prompt=system_prompt
    )

    return agent


agent = get_agent()


st.title("📝 Task Management AI")
st.caption("Manage your tasks using natural language and AI.")


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


query = st.chat_input("Ask me to create, view, update or delete tasks...")


if query:

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        placeholder = st.empty()

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

            answer = ""

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
                    answer += text_chunk
                    placeholder.markdown(answer + "▌")

            placeholder.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            st.error(f"Something went wrong: {e}")


with st.sidebar:

    st.header("About")

    st.write(
        "An AI-powered task management assistant "
        "using Streamlit, LangChain, Groq and SQLite."
    )

    if st.button("Clear Chat"):

        st.session_state.messages = []
        st.session_state.thread_id = "1"

        st.rerun()
