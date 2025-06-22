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
from graphbuilder import GraphBuilder
from langchain_core.messages import AIMessage, HumanMessage
import asyncio


# this is a simple example helps you to understand how to use tools with RAG and LLMs
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
    #1
    # This is a simple example helps you to understand how to use tools with RAG and LLMs
    # In a real application, you would replace the vector store with your actual
    print("+++++++++++++++++++Example 1++++++++++++++++++++++++")
    new_state = await APP.ainvoke({"messages": ["What is considered the best food in the world?"]})
    print(new_state["messages"][-1].content)

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    new_state = await APP.ainvoke({"messages": ["What is considered the best food in the world? is it sandwich"]})
    print(new_state["messages"][-1].content)

    print("+++++++++++++++++++Example 2++++++++++++++++++++++++")
    #2
    # GraphBuilder example also create vector store and add context creation node
    # This is a simple example helps you to understand how to use tools with RAG and LLMs
    # In a real application, you would replace the vector store with your actual implementation
    # and the context creation node would retrieve relevant documents based on the input query.
    """Initialize and run the graph with test cases"""
    builder = GraphBuilder()
    builder.initialize_components()
    app = builder.build_graph()

    # Test case 1
    print("Test case 1: General food question")
    new_state = await app.ainvoke(
        {
            "messages": [HumanMessage(content="What is considered the best food in the world?")],
            "context": ""
        })
    print("Test case 1: " + new_state["messages"][-1].content)
    
    print("\n" + "="*80 + "\n")
    
    # Test case 2
    print("Test case 2: Specific food question")
    new_state = await app.ainvoke(
        {        
            "messages": [HumanMessage(content="What is considered the best food in the world? is it sandwich")],
            "context": ""
        })
    print("Test case 2: " + new_state["messages"][-1].content)
asyncio.run(run_graph())
