# Multi-Agent Research & Synthesis System

## 1. Goal
The objective of this project is to design and implement a simple multi-agent system using **LangGraph**. The system utilizes a specialized team of AI agents to perform automated web research and synthesize findings into a professional report.

## 2. Architecture
The system follows a **Supervisor-Worker topology**, ensuring centralized coordination and iterative refinement.

*   **Supervisor (Router):** The "brain" of the system. It uses `llama-3.3-70b-versatile` (via Groq) to evaluate the state and decide the next action.
*   **Researcher Agent:** A specialized agent that utilizes the **Tavily Search API** to gather real-time information from the web.
*   **Writer Agent:** A specialized agent that synthesizes raw research data into a structured, professional summary.
*   **State Persistence (Memory):** Implemented using `MemorySaver`, allowing the agents to maintain context across multiple turns of interaction.
*   **Logic Loop (Conditional Edge):** A core feature where agents must report back to the supervisor for evaluation, allowing for iterative improvements until the task is complete.

### Architecture Graph
```mermaid
graph TD;
        __start__([<p>__start__</p>]):::first
        supervisor(supervisor)
        Researcher(Researcher)
        Writer(Writer)
        __end__([<p>__end__</p>]):::last
        Researcher --> supervisor;
        Writer --> supervisor;
        __start__ --> supervisor;
        supervisor -.-> Researcher;
        supervisor -.-> Writer;
        supervisor -. &nbsp;FINISH&nbsp; .-> __end__;
        classDef default fill:#EDF2F7,stroke:#2D3748,color:#1A202C,line-height:1.2
        classDef first fill:#C6F6D5,stroke:#276749,color:#22543D
        classDef last fill:#FED7D7,stroke:#9B2C2C,color:#742A2A