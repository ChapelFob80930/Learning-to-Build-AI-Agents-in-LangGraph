import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage 
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]] # HumanMessage and AIMessage are essentially data types in the langchain and laggraph libraries
    
llm = ChatOpenAI(model = "gpt-4.1-nano")

def process(state: AgentState)->AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])
    
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    print("CURRENT STATE: ", state["messages"])
    
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")

while user_input!="exit":
    conversation_history.append(HumanMessage(content=user_input))
    
    result = agent.invoke({"messages": conversation_history})
    
    # print(result["messages"])
    conversation_history = result["messages"]
    
    user_input = input("Enter: ")
    
with open("AI Agents\logging.txt", "w") as file:
    file.write("Your conversation log:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")
    file.write("End of conversation")

print("Conversation saved to logging.txt")
            

# -----------------------------------------------
# 💡 LangGraph Agent Flow Explanation (Quick Ref)
# -----------------------------------------------
# Essentially what's happening:
#
# 1. We define an `AgentState` which stores a list of messages.
#    These messages are of two types:
#    - `HumanMessage`: represents user input
#    - `AIMessage`: represents model responses
#
# 2. We initialize a ChatOpenAI model using LangChain.
#
# 3. The `process()` function does the main work:
#    - Receives the full message history (state["messages"])
#    - Sends it to the LLM using `llm.invoke(messages)`
#    - Receives the model's reply (`AIMessage`)
#    - Appends the reply to the message history
#    - Returns the updated state
#
# 4. We build a simple LangGraph:
#    START --> process --> END
#    This wraps the `process()` step as a graph-based agent.
#
# 5. In the main loop:
#    - Take user input
#    - Append it to `conversation_history` as a `HumanMessage`
#    - Call `agent.invoke()` to run the graph and get the AI's response
#        ↳ Inside this, the `process()` function is triggered:
#           • Sends full conversation to the LLM
#           • Gets the AI's response
#           • Appends `AIMessage` to the state
#    - Update `conversation_history` with the returned state["messages"]
#    - Repeat until user types "exit"
#
# This pattern ensures that both user and AI messages are stored and
# reused as context in every loop, enabling multi-turn conversation.

