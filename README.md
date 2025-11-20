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
├── docs/
│   ├── SRS Documentation.pdf
│   ├── Methodology.pdf
│   ├── Evaluation.pdf
│   └── Final_Report.pdf
│
├── .gitignore
├── LICENSE
├── README.md
├── cybersecurity_rag.py
├── pdf_to_text.py
└── requirements.txt
```

---

Installation

1. Clone the repository

```bash
git clone https://github.com/your-repo/cybersecurity-rag-capstone.git
cd cybersecurity-rag-capstone
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Add your OpenAI API key
   Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
```

---

Running the Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

This will open the application in your browser.

---

Technologies Used

* Python
* Streamlit
* ChromaDB
* OpenAI API
* python-dotenv
* NLP preprocessing libraries (NLTK, spaCy, etc.)

---

Team Members

| Name                   | Role                                     |
| ---------------------- | ---------------------------------------- |
| Mufaro Muwirimi    | RAG Pipeline Security Engineer           |
| Tadiwa Hukuimwe    | Lead AI & Application Security Developer |
| Takudzwa Mambosasa | Secure Data & Infrastructure Engineer    |

All team members are Cybersecurity seniors at **Southeast Missouri State University.



