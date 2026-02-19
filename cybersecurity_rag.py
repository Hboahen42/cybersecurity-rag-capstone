"""Streamlit app styled to match the aesthetic of the provided Lovable screenshot.

This script wraps the original Cybersecurity RAG demo in a custom layout
and CSS so that it better reflects a modern gradient hero design.
The functionality of the retrieval-augmented generation remains unchanged –
users can initialize a local knowledge base from documents, ask questions
with or without retrieval/reranking, and compare simple overlap metrics
between the generated answer and the retrieved context.
"""



import os
import base64
import pickle
import hashlib
from typing import List

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from openai import OpenAI


# ----------------------------------------------------------------------
# Cached model loading (prevents slow reloading)
# ----------------------------------------------------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    return embedder, reranker


# ----------------------------------------------------------------------
# Utility functions for loading and chunking local documents
# ----------------------------------------------------------------------
def chunk_text(text: str, max_length: int = 1500) -> List[str]:
    """Split text into chunks of roughly max_length characters.

    Preserves paragraph boundaries. A new chunk starts whenever adding
    another paragraph would exceed the maximum length. Trailing
    whitespace is stripped from each chunk.
    """
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
        current_chunk += para + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def get_docs_hash(docs: List[str]) -> str:
    joined = "".join(docs)
    return hashlib.md5(joined.encode()).hexdigest()


class LocalDocLoader:
    """Utility to load and chunk local documents from a directory."""

    def __init__(self, local_data_path: str = "data_sources") -> None:
        self.local_data_path = local_data_path

    def load(self) -> List[str]:
        """Load all files in the directory and return their chunks."""
        documents: List[str] = []
        if not os.path.exists(self.local_data_path):
            return documents

        for filename in os.listdir(self.local_data_path):
            filepath = os.path.join(self.local_data_path, filename)
            if not os.path.isfile(filepath):
                continue
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            for chunk in chunk_text(content, max_length=1500):
                if chunk:
                    documents.append(chunk)

        return documents


# ----------------------------------------------------------------------
# RAG pipeline definition
# ----------------------------------------------------------------------
class RAGPipeline:
    """Pipeline encapsulating embedding, reranking and generation logic."""

    def __init__(self) -> None:
        self.embedder, self.reranker = load_models()
        self.documents: List[str] = []
        self.doc_embeddings = None
        self.client = OpenAI()

    def load_knowledge_base(self, docs: List[str]) -> None:
        docs_hash = get_docs_hash(docs)
        cache_file = f"embeddings_cache_{docs_hash}.pkl"
        
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.documents, self.doc_embeddings = pickle.load(f)
                return

        self.documents = docs
        self.doc_embeddings = self.embedder.encode(docs, convert_to_tensor=True)
        
        with open(cache_file, "wb") as f:
            pickle.dump((self.documents, self.doc_embeddings), f)


    def retrieve_and_rerank(
        self, query: str, top_k_retrieve: int = 10, top_k_rerank: int = 3
    ) -> List[str]:
        """Retrieve and rerank documents for a given query."""
        if not self.documents or self.doc_embeddings is None:
            return []

        query_emb = self.embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_emb, self.doc_embeddings)[0]
        top_results = cos_scores.topk(k=min(top_k_retrieve, len(self.documents)))
        candidates = [self.documents[idx] for idx in top_results[1]]

        rerank_scores = self.reranker.predict([(query, doc) for doc in candidates])
        final_ranking = sorted(
            zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True
        )
        return [doc for doc, _ in final_ranking[:top_k_rerank]]

    def generate_qa(self, query: str, context_docs: List[str]) -> str:
        """Generate an answer given a query and optional context documents."""
        context = "\n\n".join(context_docs)
        prompt = f"""
Use the following context to answer the cybersecurity question below.

Context:
{context}

Question: {query}

Instructions:
- Give a practical, well-supported answer, using the context above when possible.
Answer:
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cybersecurity teaching assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:  # pragma: no cover - defensive
            return (
                f"**Error generating response:**\n{e}\n\nIf you see an API version error, "
                "ensure you have openai-python >=1.0.0 and use "
                "`client.chat.completions.create()`."
            )


# ----------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    """Normalize and tokenize text into lowercase words."""
    import re

    cleaned = re.sub(r"[^\w]", " ", text.lower())
    tokens = re.split(r"\s+", cleaned)
    return [t for t in tokens if t]


def compute_token_overlap(context: str, answer: str) -> float:
    """Compute percentage of answer tokens that appear in the context."""
    context_tokens = set(_tokenize(context))
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0
    overlap = [t for t in answer_tokens if t in context_tokens]
    return (len(overlap) / len(answer_tokens)) * 100.0


def compute_bigram_f1(context: str, answer: str) -> float:
    """Compute bigram F1 between context and answer."""

    def bigrams(tokens: List[str]):
        return list(zip(tokens, tokens[1:])) if len(tokens) >= 2 else []

    context_tokens = _tokenize(context)
    answer_tokens = _tokenize(answer)
    ctx_bigrams = set(bigrams(context_tokens))
    ans_bigrams = set(bigrams(answer_tokens))

    if not ans_bigrams or not ctx_bigrams:
        return 0.0

    overlap = ctx_bigrams & ans_bigrams
    precision = len(overlap) / len(ans_bigrams) if ans_bigrams else 0.0
    recall = len(overlap) / len(ctx_bigrams) if ctx_bigrams else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1 * 100.0


def compute_sentence_attribution(context: str, answer: str) -> float:
    """Compute fraction of answer sentences that share tokens with the context."""
    import re

    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
    if not sentences:
        return 0.0

    context_tokens = set(_tokenize(context))
    attributed = sum(1 for s in sentences if set(_tokenize(s)) & context_tokens)
    return (attributed / len(sentences)) * 100.0


# ----------------------------------------------------------------------
# Plotly gauges
# ----------------------------------------------------------------------
def build_gauge_chart(value: float, title: str, color: str):
    """
    Create a Plotly gauge chart with custom styling for dark theme.

    In addition to the provided colour parameter, this function
    dynamically sets the bar colour based on the current value
    (0–100). Low values are tinted red, mid values orange/yellow and
    high values green. This helps board viewers quickly interpret
    whether a metric is poor, average or strong.
    """
    # Determine a colour gradient based on value if colour is None
    if not color:
        # Map 0-100 to red→yellow→green
        if value < 33:
            color = "#ff5f5f"  # red
        elif value < 66:
            color = "#ffda5f"  # yellow
        else:
            color = "#5fff8a"  # green

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 28, "color": "#ffffff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#6a7dc0", "tickwidth": 1},
                "bar": {"color": color},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "#2e3c66",
                "borderwidth": 2,
            },
            title={"text": title, "font": {"size": 15, "color": "#aebfff"}},
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=10, r=10, t=50, b=0),
    )
    return fig


def build_semi_gauge(value: float, title: str, color: str):
    """Wrapper used in your existing code – reuse the same gauge style."""
    return build_gauge_chart(value, title, color)


# ----------------------------------------------------------------------
# Main Streamlit app
# ----------------------------------------------------------------------
def main() -> None:
    """Streamlit application entrypoint with improved UI/UX and robust logic."""
    load_dotenv()

    st.set_page_config(
        page_title="Cybersecurity RAG Demo",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ------------------------------------------------------------------
    # CSS injection – YOUR PNG + radial circular glow (Lovable-style)
    # ------------------------------------------------------------------
    # Path: /Users/takudzwa/Downloads/General/cyberecurity-rag-transfer/assets/cyber_bg.png
    gradient_path = os.path.join(
        os.path.dirname(__file__), "assets", "cyber_bg.png"
    )

    try:
        with open(gradient_path, "rb") as img_file:
            gradient_b64 = base64.b64encode(img_file.read()).decode()
    except Exception:
        gradient_b64 = ""

    st.markdown(
        f"""
        <style>
        /* Page background: your PNG + radial darkening */
        .stApp {{
            background:
                radial-gradient(
                    circle at center,
                    rgba(0, 0, 0, 0.0) 0%,
                    rgba(0, 0, 0, 0.45) 45%,
                    rgba(0, 0, 0, 0.88) 80%,
                    #020208 100%
                ),
                url("data:image/png;base64,{gradient_b64}") center/cover no-repeat fixed;
            background-attachment: fixed;
            color: #ffffff !important;
        }}

        /* Container that centers the main content and sets a max width */
        .main-container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 0 1rem;
        }}

        /* Hero section: large gradient text and subtitle */
        .hero {{
            text-align: center;
            padding: 3rem 0 2rem 0;
        }}
        .hero-title {{
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #47c2ff, #d261e9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        .hero-subtitle {{
            font-size: 1.2rem;
            color: #cfd9f7;
            margin-bottom: 1rem;
        }}
        .hero-badge {{
            display: inline-block;
            background: linear-gradient(90deg, #46c0ff, #9d4ef9);
            color: #ffffff;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            margin-bottom: 1rem;
        }}

        /* Status strip: small notification area below hero */
        .status-strip {{
            text-align: center;
            padding: 0.5rem 1rem;
            margin-bottom: 1.5rem;
            border-radius: 8px;
            background-color: rgba(70,192,255,0.15);
            color: #a9c9ff;
            font-size: 0.85rem;
        }}

        /* Chat input wrapper */
        .chat-input {{
            margin-bottom: 1rem;
        }}
        /* Chat input: pill shape with gradient border */
        .chat-input > div > div {{
            border-radius: 25px !important;
            border: 2px solid transparent !important;
            background: linear-gradient(#0d1a3a, #0d1a3a) padding-box,
                        linear-gradient(90deg, #46c0ff, #9d4ef9) border-box !important;
            box-shadow: 0 0 8px rgba(70,192,255,0.3),
                        0 0 12px rgba(157,78,249,0.2);
        }}
        .chat-input > div > div > input {{
            padding: 0.6rem 1rem !important;
            font-size: 1rem !important;
            color: #ffffff !important;
            background-color: transparent !important;
        }}

        /* Buttons: neon gradient */
        button[data-testid="baseButton"] {{
            border-radius: 25px !important;
            border: none !important;
            padding: 0.5rem 1.5rem !important;
            font-weight: 600 !important;
            color: #ffffff !important;
            background: linear-gradient(90deg, #46c0ff, #9d4ef9) !important;
            box-shadow: 0 0 12px rgba(70,192,255,0.5),
                        0 0 20px rgba(157,78,249,0.4);
            transition: background 0.3s ease, box-shadow 0.3s ease;
        }}
        button[data-testid="baseButton"]:hover {{
            background: linear-gradient(90deg, #58d4ff, #b36bfd) !important;
            box-shadow: 0 0 16px rgba(70,192,255,0.7),
                        0 0 24px rgba(179,107,253,0.6);
        }}
        button[data-testid="baseButton"]:disabled {{
            opacity: 0.4 !important;
            cursor: not-allowed !important;
            box-shadow: none !important;
        }}

        /* Answer card styling */
        .answer-card {{
            background-color: rgba(255,255,255,0.05);
            border: 1px solid rgba(111,199,255,0.15);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}

        /* Expander styling for contexts */
        .stExpander {{
            background-color: rgba(255,255,255,0.05) !important;
            border: 1px solid rgba(111,199,255,0.15) !important;
            border-radius: 10px !important;
        }}
        .stExpander > summary {{
            color: #6fc7ff !important;
            font-weight: 600 !important;
        }}

        /* Table styling */
        .stTable th, .stTable td {{
            color: #dee6f8 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------------
    if "rag_pipeline" not in st.session_state:
        st.session_state["rag_pipeline"] = None
    if "knowledge_loaded" not in st.session_state:
        st.session_state["knowledge_loaded"] = False
    # A history list to track past queries and their results. Each entry
    # stores the question, mode (RAG/ChatGPT), answer text, context, and
    # metrics. This enables a summary table for board viewers to
    # inspect past interactions at a glance.
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # ------------------------------------------------------------------
    # Hero section
    # ------------------------------------------------------------------
    st.markdown(
        "<div class='hero'>"
        "<div class='hero-badge'>Cybersecurity RAG Demo</div>"
        "<div class='hero-title'>Cybersecurity RAG in Action</div>"
        "<div class='hero-subtitle'>Ask cybersecurity questions, pull context from real docs, and measure how RAG changes the answer.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Status strip
    if st.session_state.get("knowledge_loaded") and st.session_state["rag_pipeline"]:
        doc_count = len(st.session_state["rag_pipeline"].documents)
        status_text = f"RAG initialized with {doc_count} chunks."
    else:
        status_text = "RAG not yet initialized."
    st.markdown(f"<div class='status-strip'>{status_text}</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Main content container
    # ------------------------------------------------------------------
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    # Chat input
    st.markdown("<div class='chat-input'>", unsafe_allow_html=True)
    user_query = st.text_input(
        label="Ask a cybersecurity question:",
        value="",
        key="cyber_query_input",
        placeholder="Ask a cybersecurity question...",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Retrieval parameter controls
    # Expose controls for the number of documents to retrieve and the
    # number to rerank. These sliders help boardroom users explore
    # different trade‑offs between recall and precision. Defaults are
    # stored in session state to persist across interactions.
    with st.expander("Advanced settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state["top_k_retrieve"] = st.number_input(
                "Retrieve top N docs",
                min_value=1,
                max_value=20,
                value=st.session_state.get("top_k_retrieve", 10),
                step=1,
            )
        with col2:
            st.session_state["top_k_rerank"] = st.number_input(
                "Rerank top M docs",
                min_value=1,
                max_value=10,
                value=st.session_state.get("top_k_rerank", 3),
                step=1,
            )

    # Button row
    col_init, col_rag, col_chatgpt, col_compare = st.columns(4)

    with col_init:
        if st.button(
            "Initialize RAG",
            disabled=False,
            help="Load local documents into the RAG pipeline",
        ):
            loader = LocalDocLoader()
            docs = loader.load()
            # Incorporate additional chunks uploaded via sidebar
            uploaded_chunks = st.session_state.get("uploaded_docs", [])
            if uploaded_chunks:
                docs.extend(uploaded_chunks)
            rag_pipeline = RAGPipeline()
            rag_pipeline.load_knowledge_base(docs)
            st.session_state["rag_pipeline"] = rag_pipeline
            st.session_state["knowledge_loaded"] = True
            st.success(f"Knowledge base loaded with {len(docs)} chunks (including {len(uploaded_chunks)} custom chunks).")

    with col_rag:
        rag_disabled = not (st.session_state.get("knowledge_loaded") and user_query)
        if st.button(
            "Ask with RAG + Rerank",
            disabled=rag_disabled,
            help="Generate an answer using retrieval and reranking",
        ):
            rag = st.session_state["rag_pipeline"]
            # Allow users to tweak retrieval parameters using session
            # values; these controls will be added below the input. Use
            # defaults if none specified.
            top_k_retrieve = st.session_state.get("top_k_retrieve", 10)
            top_k_rerank = st.session_state.get("top_k_rerank", 3)
            context_docs = rag.retrieve_and_rerank(
                user_query, top_k_retrieve=top_k_retrieve, top_k_rerank=top_k_rerank
            )
            rag_answer = rag.generate_qa(user_query, context_docs)
            context_text = "\n\n".join(context_docs)

            rag_tok_overlap = compute_token_overlap(context_text, rag_answer)
            rag_bigram_f1 = compute_bigram_f1(context_text, rag_answer)
            rag_sent_attr = compute_sentence_attribution(context_text, rag_answer)

            st.session_state["last_rag"] = {
                "answer": rag_answer,
                "context": context_text,
                "metrics": (rag_tok_overlap, rag_bigram_f1, rag_sent_attr),
            }

            # Record to history
            st.session_state["history"].append(
                {
                    "question": user_query,
                    "mode": "RAG + Rerank",
                    "answer": rag_answer,
                    "context": context_text,
                    "metrics": (rag_tok_overlap, rag_bigram_f1, rag_sent_attr),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

    with col_chatgpt:
        chat_disabled = not user_query
        if st.button(
            "Ask ChatGPT Only",
            disabled=chat_disabled,
            help="Generate an answer using ChatGPT without retrieval",
        ):
            rag = st.session_state.get("rag_pipeline")
            if rag is None:
                rag = RAGPipeline()
            chatgpt_answer = rag.generate_qa(user_query, context_docs=[])

            prev_context = st.session_state.get("last_rag", {}).get("context", "")
            if prev_context:
                chatgpt_tok_overlap = compute_token_overlap(prev_context, chatgpt_answer)
                chatgpt_bigram_f1 = compute_bigram_f1(prev_context, chatgpt_answer)
                chatgpt_sent_attr = compute_sentence_attribution(
                    prev_context, chatgpt_answer
                )
            else:
                chatgpt_tok_overlap = chatgpt_bigram_f1 = chatgpt_sent_attr = 0.0

            st.session_state["last_chatgpt"] = {
                "answer": chatgpt_answer,
                "context": prev_context,
                "metrics": (
                    chatgpt_tok_overlap,
                    chatgpt_bigram_f1,
                    chatgpt_sent_attr,
                ),
            }

            # Record to history
            st.session_state["history"].append(
                {
                    "question": user_query,
                    "mode": "ChatGPT Only",
                    "answer": chatgpt_answer,
                    "context": prev_context,
                    "metrics": (
                        chatgpt_tok_overlap,
                        chatgpt_bigram_f1,
                        chatgpt_sent_attr,
                    ),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

    with col_compare:

        compare_disabled = not (
            "last_rag" in st.session_state and "last_chatgpt" in st.session_state
            )
        
        if st.button(
            "Compare",
            disabled=compare_disabled,
            help="Compare the retrieval metrics of the last RAG and ChatGPT runs",
            ):
            st.session_state["show_compare"] = True

   


    # ------------------------------------------------------------------
    # Display answers
    # ------------------------------------------------------------------
    if "last_rag" in st.session_state:
        rag_res = st.session_state["last_rag"]
        st.markdown("<div class='answer-card'>", unsafe_allow_html=True)
        st.markdown("### RAG + Rerank Answer")
        st.markdown(rag_res["answer"])
        mcols = st.columns(3)
        mvals = rag_res["metrics"]


        mcols[0].plotly_chart(
            build_semi_gauge(mvals[0], "Token-Overlap", color=""),
            use_container_width=True,
            key="rag_token_overlap",
            
        )


        mcols[1].plotly_chart(
            build_semi_gauge(mvals[1], "Bigram F1", color=""),
            use_container_width=True,
            key="rag_bigram_f1",
        )


        mcols[2].plotly_chart(
            build_semi_gauge(mvals[2], "Sent Attribution", color=""),
            use_container_width=True,
            key="rag_sent_attr",
            
        )
        
        
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Top Supporting Chunks/Retrieved Contexts")
        context_docs = rag_res["context"].split("\n\n") if rag_res["context"] else []
        for i, doc in enumerate(context_docs, 1):
            with st.expander(f"Context {i}"):
                st.write(doc)

    if "last_chatgpt" in st.session_state:
        chat_res = st.session_state["last_chatgpt"]
        st.markdown("<div class='answer-card'>", unsafe_allow_html=True)
        st.markdown("### ChatGPT Only Answer")
        st.markdown(chat_res["answer"])
        mcols = st.columns(3)
        mvals = chat_res["metrics"]
        token_overlap, bigram_f1, sent_attr = mvals
        mcols[0].plotly_chart(
            build_semi_gauge(token_overlap, "Token-Overlap", color=""),
            use_container_width=True,
        )
        mcols[1].plotly_chart(
            build_semi_gauge(bigram_f1, "Bigram F1", color=""),
            use_container_width=True,
        )
        mcols[2].plotly_chart(
            build_semi_gauge(sent_attr, "Sent Attribution", color=""),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    
    
    
    # ------------------------------------------------------------------
    # Full Width Comparison Section
    # ------------------------------------------------------------------

    if (
        st.session_state.get("show_compare")
        and "last_rag" in st.session_state
        and "last_chatgpt" in st.session_state
    ):
        
        st.markdown("---")
        st.markdown("## Retrieval Metrics Comparison")
        
        
        rag_metrics = st.session_state["last_rag"]["metrics"]
        chatgpt_metrics = st.session_state["last_chatgpt"]["metrics"]

        comparison_df = pd.DataFrame(
            {
                "Metric": [
                     "Token-Overlap %",

                     "Bigram F1",

                     "Sentence Attribution",
                ],


                "RAG + Rerank": [
                    f"{rag_metrics[0]:.1f}%",
                    f"{rag_metrics[1]:.1f}%",
                    f"{rag_metrics[2]:.1f}%",

                ],
                
                "ChatGPT Only": [
                    f"{chatgpt_metrics[0]:.1f}%",
                    f"{chatgpt_metrics[1]:.1f}%",
                    f"{chatgpt_metrics[2]:.1f}%",
                ],
            }
        )

        st.dataframe(comparison_df, use_container_width=True)
    
    
    
    
    
    # ------------------------------------------------------------------
    # History & summary
    # ------------------------------------------------------------------
    history = st.session_state.get("history", [])
    if history:
        st.markdown("### Query History & Summary")
        # Build a summary table
        data = {
            "Question": [h["question"] for h in history],
            "Mode": [h["mode"] for h in history],
            "Token-Overlap %": [f"{h['metrics'][0]:.1f}" for h in history],
            "Bigram F1": [f"{h['metrics'][1]:.1f}" for h in history],
            "Sent Attribution %": [f"{h['metrics'][2]:.1f}" for h in history],
        }
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        # Display average metrics for RAG and ChatGPT separately
        rag_entries = [h for h in history if h["mode"] == "RAG + Rerank"]
        chat_entries = [h for h in history if h["mode"] == "ChatGPT Only"]
        def avg_metric(entries, idx):
            return sum(h["metrics"][idx] for h in entries) / len(entries) if entries else 0.0
        col_summary1, col_summary2 = st.columns(2)
        with col_summary1:
            st.markdown("**Average metrics – RAG + Rerank**")
            st.markdown(f"Token-Overlap: {avg_metric(rag_entries,0):.1f}%")
            st.markdown(f"Bigram F1: {avg_metric(rag_entries,1):.1f}%")
            st.markdown(f"Sent Attribution: {avg_metric(rag_entries,2):.1f}%")
        with col_summary2:
            st.markdown("**Average metrics – ChatGPT Only**")
            st.markdown(f"Token-Overlap: {avg_metric(chat_entries,0):.1f}%")
            st.markdown(f"Bigram F1: {avg_metric(chat_entries,1):.1f}%")
            st.markdown(f"Sent Attribution: {avg_metric(chat_entries,2):.1f}%")
        # Export history as CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download history as CSV",
            data=csv_data,
            file_name="rag_demo_history.csv",
            mime="text/csv",
        )

    # Close container
    st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    with st.sidebar:
        st.info(
            "This demo lets you compare a retrieval-augmented answer with a plain ChatGPT answer.\n\n"
            "1. **Initialize RAG** to load local documents.\n"
            "2. Enter your cybersecurity question.\n"
            "3. Try **Ask with RAG + Rerank** or **Ask ChatGPT Only**.\n"
            "4. Use **Compare** to see metric differences once you have both answers.\n\n"
            "If you see API errors, ensure you have openai-python >=1.0.0 and the new "
            "`client.chat.completions.create()` syntax."
        )

        # --------------------------------------------------------------
        # Custom document uploader
        # --------------------------------------------------------------
        uploaded = st.file_uploader(
            "Upload additional docs (txt)",
            type=["txt", "md"],
            accept_multiple_files=True,
            help="Optional: supply your own documents for the demo. Only plain text files are supported."
        )
        additional_chunks: List[str] = []
        if uploaded:
            for uf in uploaded:
                try:
                    content = uf.read().decode("utf-8", errors="ignore")
                    for ch in chunk_text(content, max_length=1500):
                        if ch:
                            additional_chunks.append(ch)
                except Exception:
                    continue
        # Persist uploaded docs in session state for use during initialization
        st.session_state["uploaded_docs"] = additional_chunks


if __name__ == "__main__":
    main()



