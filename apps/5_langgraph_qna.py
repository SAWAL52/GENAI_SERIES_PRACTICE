from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated,List


class ChatState(BaseModel):
    messages:Annotated[List,add_messages]

llm=ChatGroq(model="openai/gpt-oss-120b")

def chatBotNode(state:ChatState) -> ChatState:
    response=llm.invoke(state.messages)
    state.messages=[response]
    return state

memory=InMemorySaver()

graph=StateGraph(ChatState)
graph.add_node("chatBot",chatBotNode)

graph.add_edge(START,"chatBot")
graph.add_edge("chatBot",END)

graph=graph.compile(checkpointer=memory)

config={"configurable":{"thread_id":"my-bot-1"}}

while True:
    query=input("User: ")
    if query.lower() in ["quit","exit","end"]:
        print("Bye bye!")
        break


    response=graph.invoke({"messages":[{"role":"user","content":query}]}, config)

    answer=response["messages"][-1].content
    print("AI: ",answer)

