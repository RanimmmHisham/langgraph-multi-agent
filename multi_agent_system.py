import os
import operator
from typing import Annotated, List, Literal, TypedDict
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 1. SETUP KEYS (Use environment variables for security)
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# 2. DEFINE STATE
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str

# 3. INITIALIZE MODELS & TOOLS
llm = ChatGroq(model="llama-3.3-70b-versatile")
search_tool = TavilySearch(max_results=3)

# 4. DEFINE THE SUPERVISOR (Router)
members = ["Researcher", "Writer"]
options = ["FINISH"] + members

class RouteResponse(BaseModel):
    next: Literal[*options]

supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a supervisor managing a research team: {members}. "
               "Based on the conversation, decide who should act next. "
               "If you have a final summary that satisfies the user, respond with FINISH."),
    ("placeholder", "{messages}"),
]).partial(members=", ".join(members))

def supervisor_node(state: AgentState):
    chain = supervisor_prompt | llm.with_structured_output(RouteResponse)
    response = chain.invoke(state)
    return {"next": response.next}

# 5. DEFINE AGENT NODES
def researcher_node(state: AgentState):
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools([search_tool])
    # Get response
    response = llm_with_tools.invoke(state["messages"])
    
    # Execute tool calls manually for this simple flow
    tool_messages = []
    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = search_tool.invoke(tool_call["args"])
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    
    return {"messages": [response] + tool_messages}

def writer_node(state: AgentState):
    # The writer synthesizes the research found in the message history
    writer_prompt = "Using the research provided in the history, write a concise, professional summary."
    response = llm.invoke(state["messages"] + [HumanMessage(content=writer_prompt)])
    return {"messages": [response]}

# 6. BUILD THE GRAPH
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Writer", writer_node)

# Add Edges (Workers always return to supervisor)
workflow.add_edge("Researcher", "supervisor")
workflow.add_edge("Writer", "supervisor")

# Add Conditional Logic
workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "Researcher": "Researcher",
        "Writer": "Writer",
        "FINISH": END
    }
)

workflow.add_edge(START, "supervisor")

# 7. COMPILE WITH MEMORY
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# ==========================================
# 8. THE EXECUTION BLOCK (PUT THIS HERE)
# ==========================================
if __name__ == "__main__":
    # Configuration for memory persistence
    config = {"configurable": {"thread_id": "graduation_test_001"}}

    # The user request
    input_msg = {
        "messages": [
            HumanMessage(content="Research the impact of AI on clinical trials and summarize it.")
        ]
    }

    print("--- Starting Multi-Agent Collaboration ---\n")
    
    # Streaming the events from the graph
    for event in graph.stream(input_msg, config):
        for node, values in event.items():
            print(f"Node '{node}' is acting...")
            # If the supervisor made a decision, print it
            if "next" in values:
                print(f"Decision: Next agent is {values['next']}")
            print("-" * 30)

# Print the Mermaid syntax to the console
# 1. Generate the Mermaid markdown string
mermaid_graph = graph.get_graph().draw_mermaid()

# 2. Print it to the console so you can copy it
print("\n--- MERMAID GRAPH CODE ---")
print(mermaid_graph)
print("--------------------------\n")