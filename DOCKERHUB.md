# Cybernet — Cybersecurity RAG Demo

A containerized Retrieval-Augmented Generation (RAG) application for cybersecurity Q&A. Ask cybersecurity questions, retrieve context from real documents, and compare how RAG changes the answer versus a standalone LLM.

## What This Image Does

This image runs a **Streamlit web application** that:

- Embeds and indexes cybersecurity documents using **Sentence-Transformers**
- Retrieves and reranks relevant context for your questions using a **CrossEncoder**
- Generates answers via **OpenAI GPT-4o-mini**, with or without retrieved context
- Displays side-by-side metrics (Token Overlap, Bigram F1, Sentence Attribution) so you can measure how retrieval improves the answer

The project documentation PDFs are pre-converted to text and baked into the image, so the knowledge base is ready to use immediately.

## Quick Start

```bash
docker pull hboahen/cybernet:latest
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key_here hboahen/cybernet:latest
```

Then open **http://localhost:8501**.

## Requirements

| Requirement        | Details                                                                                                                                   |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| **OpenAI API Key** | Required. Pass it as an environment variable (see below). You can get one at [platform.openai.com](https://platform.openai.com/api-keys). |
| **Docker**         | Docker Desktop or Docker Engine installed on your machine.                                                                                |
| **Port 8501**      | The app listens on this port by default.                                                                                                  |

## Environment Variables

| Variable                | Required | Description                                           |
|-------------------------|----------|-------------------------------------------------------|
| `OPENAI_API_KEY`        | Yes      | Your OpenAI API key for GPT-4o-mini answer generation |
| `STREAMLIT_SERVER_PORT` | No       | Override the default port (8501)                      |

### Passing your API key

**Inline:**

```bash
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... hboahen/cybernet:latest
```

**Using an env file (recommended — keeps the key out of shell history):**

Create a file called `.env`:

```
OPENAI_API_KEY=sk-...
```

```bash
docker run -p 8501:8501 --env-file .env hboahen/cybernet:latest
```

## Using Your Own Documents

The image ships with preloaded documents, but you can mount your own:

```bash
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=sk-... \
  -v /path/to/your/docs:/app/data_sources \
  hboahen/cybernet:latest
```

Place `.txt` files in the mounted directory. Click **Initialize RAG** in the app to index them.

## How to Use the App

1. Click **Initialize RAG** to load and embed the knowledge base.
2. Type a cybersecurity question in the text input.
3. Click **Ask with RAG + Rerank** to get a retrieval-augmented answer.
4. Click **Ask ChatGPT Only** to get an answer without retrieval.
5. Click **Compare** to see metric differences side by side.
6. Use **Advanced settings** to tune the number of documents retrieved and reranked.

## Image Details

| Property     | Value                                  |
|--------------|----------------------------------------|
| Base image   | `python:3.11-slim`                     |
| Exposed port | `8501`                                 |
| Entrypoint   | `streamlit run cybersecurity_rag.py`   |
| Health check | `http://localhost:8501/_stcore/health` |
| Build        | Multi-stage (CPU-only PyTorch)         |

## Source Code

[github.com/Hboahen42/cybersecurity-rag-capstone](https://github.com/Hboahen42/cybersecurity-rag-capstone)

## License

Apache-2.0