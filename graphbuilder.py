from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma, Pinecone
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import asyncio
import os

# Configuration (Choose one)
VECTOR_STORE = "chroma"  # "chroma" or "pinecone"

# Pinecone setup (only needed if using Pinecone)
if VECTOR_STORE == "pinecone":
    from pinecone import Pinecone as PineconeClient
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = "food-index"

class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str

class GraphBuilder:
    def __init__(self):
        self.graph = StateGraph(State)
        self.llm = None
        self.tools = []
        self.vectorstore = None
        self.embeddings = OllamaEmbeddings(model="llama3.2:latest", base_url="http://localhost:11434")

    def initialize_components(self):
        self._initialize_llm()
        self._initialize_tools()
        self._initialize_vectorstore()

    def _initialize_llm(self):
        self.llm = ChatOllama(model="llama3.2:latest", base_url="http://localhost:11434")

    @tool
    def get_external_data(input_data: str) -> str:
        """Retrieves additional food-related information based on input keywords.
        
        Args:
            input_data: Food item or category to search for
            
        Returns:
            str: Expanded food suggestions based on the input
        """
        if input_data.lower() in ["sandwich"]:
            return "Consider sandwich, burgers, any food which is served between bread"
        return "Consider food from all countries."
    
    def _initialize_tools(self):        
        self.tools = [self.get_external_data]

    def _initialize_vectorstore(self):
        documents = [
            Document(page_content="Italian cuisine is famous for its pasta and pizza"),
            Document(page_content="Japanese cuisine is renowned for sushi and ramen"),
            Document(page_content="Mexican cuisine is popular for tacos and burritos"),
        ]

        if VECTOR_STORE == "chroma":
            # Chroma (local persistent DB)
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings               
            )
        elif VECTOR_STORE == "pinecone":
            # Pinecone (cloud-based)
            PineconeClient(api_key=PINECONE_API_KEY)
            self.vectorstore = Pinecone.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name=PINECONE_INDEX_NAME
            )
        else:
            raise ValueError(f"Unknown vector store: {VECTOR_STORE}")

        self.vectorstore = self.vectorstore.as_retriever()

    # ... (rest of the GraphBuilder methods remain the same as previous example)
    def build_graph(self):
        """Construct and connect all graph nodes"""
        self._add_external_data_node()
        self._add_context_creation_node()
        self._add_llm_node()
        self._configure_graph_flow()
        return self.graph.compile()

    def _add_external_data_node(self):
        """Modified tool node that handles message conversion"""
        async def tool_wrapper(state: State):
            # Ensure we have an AIMessage with tool calls
            last_msg = state["messages"][-1]
            
            if not isinstance(last_msg, AIMessage):
                raise ValueError(f"Expected AIMessage, got {type(last_msg)}")
            
            if not last_msg.tool_calls:
                return {"messages": [last_msg], "context": state["context"]}
            
            # Process tool calls
            tool_messages = []
            for tool_call in last_msg.tool_calls:
                tool = next(t for t in self.tools if t.name == tool_call["name"])
                output = tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=str(output),
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"]
                    )
                )
            
            return {
                "messages": state["messages"] + tool_messages,
                "context": state["context"]
            }

        self.graph.add_node("get_external_data", RunnableLambda(tool_wrapper))

    def _add_context_creation_node(self):
        """Add node for context creation from vector store"""
        async def create_context(state: State) -> State:
            last_message = state["messages"][-1].content
            docs = await self.vectorstore.ainvoke(last_message)
            context = "\n".join([doc.page_content for doc in docs])
            return {"messages": state["messages"], "context": context}
        
        self.graph.add_node("create_context", RunnableLambda(create_context))

    def _add_llm_node(self):        
        async def call_llm(state: State) -> State:
            # Convert any string inputs to HumanMessage
            messages = [
                msg if isinstance(msg, (AIMessage, HumanMessage, ToolMessage))
                else HumanMessage(content=str(msg))
                for msg in state["messages"]
            ]
            
            llm_with_tools = self.llm.bind_tools(self.tools)
            response = await llm_with_tools.ainvoke(messages)
            
            if not isinstance(response, AIMessage):
                response = AIMessage(content=str(response))
            
            return {
                "messages": messages + [response],
                "context": state["context"]
            }

        self.graph.add_node("call_llm", RunnableLambda(call_llm))
        
    async def call_llm(self, state: State) -> State:        
        messages = state["messages"]
        
        # Ensure last message is from human
        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("Last message should be from human")
        
        llm_with_tools = self.llm.bind_tools(self.tools)
        response = await llm_with_tools.ainvoke(messages)
        
        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(response))
        
        return {"messages": messages + [response], "context": state["context"]}

    def conditional_edge(self, state: State) -> Literal['get_external_data', '__end__']:        
        last_msg = state["messages"][-1]
        
        # Check for both new and old tool call formats
        tool_calls = getattr(last_msg, 'tool_calls', None) or getattr(last_msg, 'tool_call', None)
    
        return "get_external_data" if tool_calls else "__end__"
    def _configure_graph_flow(self):        
        def conditional_edge(state: State):
            last_msg = state["messages"][-1]
            
            if not isinstance(last_msg, AIMessage):
                return "__end__"
                
            return "get_external_data" if last_msg.tool_calls else "__end__"
        
        self.graph.add_conditional_edges(
            "call_llm",
            conditional_edge
        )
        self.graph.add_edge("get_external_data", "call_llm")
        self.graph.set_entry_point("call_llm")
