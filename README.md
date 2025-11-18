# cybersecurity-rag-capstone
Cybersecurity RAG system using OpenAI embeddings and ChromaDB with secure-by-design architecture

This repository contains the implementation of our Cybersecurity capstone project: a Retrieval-Augmented Generation (RAG) system designed to evaluate and compare responses between a standard LLM (OpenAI) and a RAG-enhanced model using ChromaDB.

The system provides:
- Secure ingestion of knowledge base documents  
- A safe RAG pipeline with sanitization and validation  
- Multiple evaluation metrics (Token Overlap, Bigram F1, Sentence Attribution)  
- A Streamlit-based user interface  
- Security-focused architecture and logic  
- Logging and traceability for analysis  


Project Structure
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


## How to Install
1. Clone the repository  
2. Install dependencies:

bash
pip install -r requirements.txt

Add your API key in a .env file:
OPENAI_API_KEY=your_key_here

How to Run the Application

bash
streamlit run app.py

This will launch the interface in your browser.

Technologies Used:

Python
Streamlit
ChromaDB
OpenAI API
Python-dotenv
Standard NLP preprocessing libraries

Team Members:
Mufaro Muwirimi – RAG Pipeline Security Engineer

Tadiwa Hukuimwe – Lead AI & Application Security Developer

Takudzwa Mambosasa – Secure Data & Infrastructure Engineer

All team members are Cybersecurity seniors at Southeast Missouri State University.
