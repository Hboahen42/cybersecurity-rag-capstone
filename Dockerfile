FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by torch/sentence-transformers
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Convert PDFs to text so the RAG knowledge base is baked into the image.
# pdf_to_text.py reads from data_sources/, so copy the PDFs there first.
RUN mkdir -p data_sources && \
    cp docs/*.pdf data_sources/ 2>/dev/null; \
    python pdf_to_text.py || true

# Streamlit config: disable CORS/XSRF for container environments
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "cybersecurity_rag.py"]