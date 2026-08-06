# Digital Twin Q&A

RAG chatbot over your resume/LinkedIn/summary. Two implementations of the same bot, for comparison:

- `langchain_app.py` — plain LangChain, LCEL chain + `RunnableWithMessageHistory`
- `langgraph_app.py` — LangGraph `StateGraph` with a `retrieve` → `generate` graph and a checkpointer for memory

Both read from the same Chroma vector store and serve a Gradio chat window.

## Setup

```bash
cp .env.example .env        # then edit .env and add your OPENAI_API_KEY
source .venv/bin/activate   # venv already created with deps installed
python ingest.py            # one-time: embeds data/*.pdf + summary.txt into ./chroma_db
```

## Run

```bash
python langchain_app.py     # opens at http://127.0.0.1:7860
# or
python langgraph_app.py     # opens at http://127.0.0.1:7860
```

## Data

`data/resume.pdf`, `data/linkedin.pdf`, `data/summary.txt` — edit/replace and re-run `ingest.py` to refresh the vector store.
