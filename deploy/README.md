---
title: Bilal Digital Twin (LangGraph)
emoji: 🧑‍💻
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
---

# Bilal's Digital Twin

A RAG chatbot answering questions about Bilal's background, built with LangGraph
(`StateGraph` + `MemorySaver` checkpointer) over his resume, LinkedIn, and summary,
indexed in Chroma.

**Requires a Space secret:** `OPENAI_API_KEY` — set it in this Space's
Settings → Variables and secrets before it will respond.
