"""Digital-twin Q&A chatbot built with plain LangChain (LCEL + chat history)."""
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PERSONA_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {
        "context": (lambda x: x["question"]) | retriever | format_docs,
        "question": lambda x: x["question"],
        "history": lambda x: x["history"],
    }
    | prompt
    | llm
    | StrOutputParser()
)

session_history = ChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: session_history,
    input_messages_key="question",
    history_messages_key="history",
)


def respond(message, history):
    return chain_with_history.invoke(
        {"question": message},
        config={"configurable": {"session_id": "local"}},
    )


if __name__ == "__main__":
    gr.ChatInterface(
        respond,
        title="Bilal's Digital Twin (LangChain)",
        description="Ask me about my experience, skills, or background.",
    ).launch()
