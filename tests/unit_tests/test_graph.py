from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages # Reducer function appends messages instead of replacing
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import json

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    results: str 
@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""
    
    return a + b

@tool
def multiply(a: int, b: int):
    """This is a multiplication function that multiplies 2 numbers together"""

    return a * b

tools = [add, multiply] # a list of tools that can be used by the agent

model = ChatOllama(model="qwen3:8b", temperature=0).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content = "You are a mathematics AI assistant. Return your final answer in a single JSON object with the keys 'result' and 'explanation'."
    )
    response = model.invoke([system_prompt] + state["messages"])
    rescont = response.content.strip()
    content_result = rescont.split("</think>")[-1].strip()
    content_think = rescont.split("</think>")[0].replace("<think>", "").strip() 
    #print("THINK:", content_think)
    #print("RESULT:", content_result)
    result = "NA"
    if content_result:
        try:
            parsed = json.loads(content_result)
            result = parsed.get("result", "NA")
            global answer 
            answer = result
        except Exception as e:
            print("Error parsing JSON:", e)
            
    return {"messages": [response], "results": result}

# Conditional edge
def should_continue(state:AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # if the last message is not a ToolMessage, we end the graph
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
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

with open ("graph.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())
    
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 555555 + 666666 and then multiply the result by 777777.")]}
print_stream(app.stream(inputs, stream_mode="values"))

print("Final Answer:", answer)
