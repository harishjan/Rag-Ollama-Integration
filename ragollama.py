from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
import asyncio

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph = StateGraph(State)


@tool
def get_external_data(input_data: str):    
    "return more information based on input data"
    if input_data.lower() in ["sandwich"]:
        return "Consider sandwich, burgers, any food which is served between bread"
    else:
        return "Consider food from all countries."
    


llm = ChatOllama(model="llama3.2:latest", base_url="http://localhost:11434") 

tools = [get_external_data]
tool_node = ToolNode(tools)
graph.add_node("tool_node", tool_node)


llm_with_tools = llm.bind_tools(tools)

async def prompt_node(state: State) -> State:    
    new_message = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [new_message]}

# 4. Create the node using RunnableLambda
tool_calling_node = RunnableLambda(prompt_node)
graph.add_node("prompt_node", tool_calling_node)


def conditional_edge(state: State) -> Literal['tool_node', '__end__']:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    else:
        return "__end__"
    
graph.add_conditional_edges(
    'prompt_node',
    conditional_edge
)
graph.add_edge("tool_node", "prompt_node")
graph.set_entry_point("prompt_node")


APP = graph.compile()

async def run_graph():
    new_state = await APP.ainvoke({"messages": ["What is considered the best food in the world?"]})
    print(new_state["messages"][-1].content)

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    new_state = await APP.ainvoke({"messages": ["What is considered the best food in the world? is it sandwich"]})
    print(new_state["messages"][-1].content)

asyncio.run(run_graph())
