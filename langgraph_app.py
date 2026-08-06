"""Digital-twin Q&A chatbot built with LangGraph (StateGraph + checkpointer)."""
from pathlib import Path
from typing import Annotated, TypedDict

import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

PERSIST_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "digital_twin"

PERSONA_PROMPT = (
    "You are Bilal's digital twin, answering questions on his behalf using the "
    "context retrieved from his resume, LinkedIn, and summary. Speak in the first "
    "person as Bilal. If the context doesn't contain the answer, say you don't "
    "have that information rather than making something up.\n\nContext:\n{context}"
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory=PERSIST_DIR,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str


def retrieve(state: State):
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    return {"context": "\n\n".join(doc.page_content for doc in docs)}


def generate(state: State):
    system = SystemMessage(content=PERSONA_PROMPT.format(context=state["context"]))
    response = llm.invoke([system, *state["messages"]])
    return {"messages": [response]}


graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)
graph = graph_builder.compile(checkpointer=MemorySaver())


def respond(message, history):
    result = graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": "local"}},
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    gr.ChatInterface(
        respond,
        title="Bilal's Digital Twin (LangGraph)",
        description="Ask me about my experience, skills, or background.",
    ).launch(share=True)
