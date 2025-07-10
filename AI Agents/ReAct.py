from typing import TypedDict, Sequence, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage # The foundational class for all message types in langgraph
from langchain_core.messages import ToolMessage # Passes data back to the LLM after it calls a tool such as the content and the tool_call_id 
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# Annotated - provides additional context without affecting the type itself
# email = Annotated[str, "This has to be a valid email format"]
# print(email.__metadata__)
# output = ('This has to be a valid email format',)

# Sequence - To automatically handle the state updates for sequences such as by adding new messages to a chat history

# add_messages is a Reducer Function 
# Reducer Function:
# Rule that controls how updates from nodes are combined with the existing state.
# tells us how to merge new data into the current state

# Without a reducer, updates would have replaces the existing value entirely!

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
@tool
def add(a:int, b:int):
    """This is an addition function that adds 2 numbers together"""
    return a+b

@tool
def subtract(a:int, b:int):
    """Subtraction function"""
    return a-b

@tool
def multiplication(a:int, b:int):
    """Multiplication function"""
    return a*b

tools = [add, subtract, multiplication]

model = ChatOpenAI(model = "gpt-4.1-nano").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assisstant, please answer my query to the best of your ability.")
    response = model.invoke([system_prompt]+state["messages"])
    return {"messages":[response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue" : "tools",
        "end" : END
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
            
inputs = {"messages":[("user", "Add 40 + 12. Add 100 and 1000. Then multiply the 2 results. Also finally tell me a joke")]}
print_stream(app.stream(inputs, stream_mode="values"))
