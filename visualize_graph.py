"""Print the compiled LangGraph's structure as Mermaid, using LangGraph's built-in draw_mermaid()."""
from langgraph_app import graph

if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())
