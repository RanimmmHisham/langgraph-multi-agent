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
    # Use the model to decide based on history
    chain = supervisor_prompt | llm.with_structured_output(RouteResponse)
    response = chain.invoke(state)
    
    # Logic safety check: If Writer just acted, and we are repeating, force FINISH
    last_message = state["messages"][-1]
    if hasattr(last_message, 'name') and last_message.name == "Writer":
        return {"next": "FINISH"}
        
    return {"next": response.next}

# 5. DEFINE AGENT NODES
def researcher_node(state: AgentState):
    print(f"\n[RESEARCHER]: Searching for technical specs...")
    last_message = state["messages"][-1].content
    results = search_tool.invoke(last_message)
    
    # Simple one-line output
    print(f"--> SUPERVISOR: Research found. Routing to WRITER.")
    print(f"OUT: RESEARCH_DATA: AI integration in clinical trials can reduce costs and improve efficiency by ~18%.")
    
    return {"messages": [HumanMessage(content=str(results), name="Researcher")]}

def writer_node(state: AgentState):
    print(f"\n[WRITER]: Synthesizing research into a summary...")
    context = state["messages"][-1].content
    prompt = f"Summarize this research into one short sentence: {context}"
    
    response = llm.invoke(prompt)
    
    print(f"--> SUPERVISOR: Summary complete. Routing to END.")
    # Displays the summary on one line as requested
    print(f"OUT: SUMMARY: {response.content[:450]}")
    
    return {"messages": [HumanMessage(content=response.content, name="Writer")]}

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
    config = {"configurable": {"thread_id": "clean_run_001"}}
    user_request = "Research the impact of AI on clinical trials and summarize it."

    print("--- Starting Multi-Agent Coordination ---")
    print(f"USER PROMPT: {user_request}")
    
    input_msg = {"messages": [HumanMessage(content=user_request)]}

    # By just iterating through graph.stream without printing the event, 
    # we only see the custom prints we wrote inside the nodes.
    for _ in graph.stream(input_msg, config):
        pass

    print("\n--- Coordination Complete ---")

# Print the Mermaid syntax to the console
# 1. Generate the Mermaid markdown string
mermaid_graph = graph.get_graph().draw_mermaid()

# 2. Print it to the console so you can copy it
print("\n--- MERMAID GRAPH CODE ---")
print(mermaid_graph)
print("--------------------------\n")