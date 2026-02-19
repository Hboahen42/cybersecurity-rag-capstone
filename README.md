Cybersecurity RAG System (Secure-by-Design)

A Retrieval-Augmented Generation (RAG) system using OpenAI embeddings and ChromaDB, built with a security-first architecture.

This project is the implementation of our Cybersecurity Capstone, focused on evaluating and comparing responses from:

* A standard LLM (OpenAI)
* A RAG-enhanced model using ChromaDB

---

Key Features

* Secure ingestion of knowledge-base documents
* Hardened RAG pipeline with sanitization & validation
* Multiple evaluation metrics:

  * Token Overlap
  * Bigram F1
  * Sentence Attribution
* Streamlit-based interface for testing and evaluation
* Secure-by-design architecture for all stages of the pipeline
* Logging & traceability for reproducible analysis

---

Project Structure

```
cybersecurity-rag-capstone/
│
├── .github/
│   └── workflows/
│       └── docker-build-and-push.yml   # CI/CD pipeline for Docker Hub
│
├── docs/
│   ├── SRS Documentation.pdf
│   ├── Methodology.pdf
│   ├── Evaluation Results.pdf
│   └── Project Report.pdf
│
├── .dockerignore          # Files excluded from the Docker image
├── .gitignore
├── Dockerfile             # Multi-stage Docker build definition
├── DOCKERHUB.md           # Docker Hub description / quick-start for container users
├── LICENSE
├── README.md
├── cybersecurity_rag.py   # Main Streamlit RAG application
├── pdf_to_text.py         # Utility to convert PDFs to plain text
└── requirements.txt
```

---

Installation

### Option 1 — Run locally

1. Clone the repository

```bash
git clone https://github.com/Hboahen42/cybersecurity-rag-capstone.git
cd cybersecurity-rag-capstone
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Add your OpenAI API key — create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
```

4. Launch the Streamlit interface:

```bash
streamlit run cybersecurity_rag.py
```

### Option 2 — Run with Docker

Pull the pre-built image from Docker Hub and run it:

```bash
docker pull <your-dockerhub-username>/cybernet:latest
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key_here <your-dockerhub-username>/cybernet:latest
```

Then open **http://localhost:8501** in your browser.

> The Docker image comes with the project PDFs already converted to text, so **Initialize RAG** works out of the box — no volume mounts needed.

See [`DOCKERHUB.md`](DOCKERHUB.md) for full Docker usage details.

---

Docker & Deployment
---

### What was added

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build that containerizes the Streamlit app with CPU-only PyTorch to keep the image small |
| `.dockerignore` | Excludes `.git`, `.idea`, cache files, secrets, and other non-essential files from the image |
| `.github/workflows/docker-build-and-push.yml` | GitHub Actions workflow that automatically builds and pushes the image to Docker Hub on every push to `main` |
| `DOCKERHUB.md` | Documentation for Docker Hub users explaining what the image is and how to run it |

### How it works

1. **On every push to `main`**, the GitHub Actions workflow triggers automatically.
2. The workflow logs into Docker Hub using repository secrets (`DOCKER_USERNAME` and `DOCKER_PASSWORD`).
3. It builds the Docker image using a **multi-stage Dockerfile**:
   - **Builder stage** — installs `build-essential` and all Python dependencies (CPU-only PyTorch from the dedicated index to save ~1.5 GB).
   - **Runtime stage** — copies only the installed packages and application code into a clean `python:3.11-slim` image.
   - During the build, PDFs in `docs/` are converted to text files in `data_sources/` so the knowledge base is baked into the image.
4. The image is tagged with `latest`, the branch name, and a timestamped production tag, then pushed to Docker Hub.
5. GitHub Actions layer caching (`type=gha`) is used so subsequent builds are faster.

### Repository secrets required

These must be set in **GitHub > Settings > Secrets and variables > Actions**:

| Secret | Value |
|--------|-------|
| `DOCKER_USERNAME` | Your Docker Hub username |
| `DOCKER_PASSWORD` | A Docker Hub [access token](https://hub.docker.com/settings/security) |

---

Technologies Used

* Python
* Streamlit
* Sentence-Transformers & CrossEncoder (local embedding and reranking)
* OpenAI API (GPT-4o-mini for answer generation)
* PyTorch (CPU)
* Plotly (metric gauge charts)
* pdfplumber (PDF to text conversion)
* Docker & GitHub Actions (CI/CD)

---

Team Members

| Name                   | Role                                     |
| ---------------------- | ---------------------------------------- |
| Mufaro Muwirimi    | RAG Pipeline Security Engineer           |
| Tadiwa Hukuimwe    | Lead AI & Application Security Developer |
| Takudzwa Mambosasa | Secure Data & Infrastructure Engineer    |

All team members are Cybersecurity seniors at **Southeast Missouri State University.



