from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

llm = init_chat_model("openai:gpt-4.1-nano")

class AgentState(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    
graph = StateGraph(AgentState)

def chatbot(state: AgentState)->AgentState:
    return {"messages": [llm.invoke(state["messages"])]}

graph.add_node("chatbot", chatbot)
graph.add_edge(START,"chatbot")
graph.add_edge("chatbot", END)

agent = graph.compile()

user_input = input("Enter a message: ")
state = agent.invoke({"messages":[{"role":"user", "content":user_input}]})

print(state['messages'][-1].content)