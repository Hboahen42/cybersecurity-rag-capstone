import os
import json
from typing import List
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from openai import OpenAI   # ✅ NEW API import


def chunk_text(text, max_length=1500):
    paragraphs = text.split('\n\n')
    chunks, current_chunk = [], ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
        current_chunk += para + "\n\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


class LocalDocLoader:
    def __init__(self, local_data_path: str = "data_sources") -> None:
        self.local_data_path = local_data_path

    def load(self) -> List[str]:
        documents = []
        for filename in os.listdir(self.local_data_path):
            filepath = os.path.join(self.local_data_path, filename)
            if not os.path.isfile(filepath): 
                continue
            # Fixed here: ignore decoding errors
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            for chunk in chunk_text(content, max_length=1500):
                if chunk: 
                    documents.append(chunk)
        return documents


class RAGPipeline:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
        self.documents = []
        self.doc_embeddings = None
        self.client = OpenAI()   # ✅ NEW client instance

    def load_knowledge_base(self, docs: List[str]):
        self.documents = docs
        self.doc_embeddings = self.embedder.encode(docs, convert_to_tensor=True)

    def retrieve_and_rerank(self, query, top_k_retrieve=10, top_k_rerank=3):
        query_emb = self.embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_emb, self.doc_embeddings)[0]
        top_results = cos_scores.topk(k=min(top_k_retrieve, len(self.documents)))
        candidates = [self.documents[idx] for idx in top_results[1]]
        rerank_scores = self.reranker.predict([(query, doc) for doc in candidates])
        final_ranking = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in final_ranking[:top_k_rerank]]

    def generate_qa(self, query, context_docs):
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
            # ✅ Updated for new OpenAI API (v1+)
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # or "gpt-3.5-turbo" if preferred
                messages=[
                    {"role": "system", "content": "You are a cybersecurity teaching assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"**Error generating response:**\n{e}\n\nIf you see an API version error, ensure you have openai-python >=1.0.0+ and use `client.chat.completions.create()`."


def _tokenize(text: str) -> List[str]:
    import re
    cleaned = re.sub(r"[^\w]", " ", text.lower())
    tokens = re.split(r"\s+", cleaned)
    return [t for t in tokens if t]


def compute_token_overlap(context: str, answer: str) -> float:
    context_tokens = set(_tokenize(context))
    answer_tokens = _tokenize(answer)
    if not answer_tokens: 
        return 0.0
    overlap = [t for t in answer_tokens if t in context_tokens]
    return (len(overlap) / len(answer_tokens)) * 100.0


def compute_bigram_f1(context: str, answer: str) -> float:
    def bigrams(tokens): return list(zip(tokens, tokens[1:])) if len(tokens) >= 2 else []
    context_tokens = _tokenize(context)
    answer_tokens = _tokenize(answer)
    ctx_bigrams = set(bigrams(context_tokens))
    ans_bigrams = set(bigrams(answer_tokens))
    if not ans_bigrams or not ctx_bigrams: 
        return 0.0
    overlap = ctx_bigrams & ans_bigrams
    precision = len(overlap) / len(ans_bigrams) if ans_bigrams else 0
    recall = len(overlap) / len(ctx_bigrams) if ctx_bigrams else 0
    if precision + recall == 0: 
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1 * 100.0


def compute_sentence_attribution(context: str, answer: str) -> float:
    import re
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
    if not sentences: 
        return 0.0
    context_tokens = set(_tokenize(context))
    attributed = sum(1 for s in sentences if set(_tokenize(s)) & context_tokens)
    return (attributed / len(sentences)) * 100.0


def build_gauge_chart(value, title, color):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number", value=value,
            number={"suffix": "%", "font": {"size": 28, "color": "#fff"}},
            gauge={"axis": {"range": [0,100]}, "bar": {"color": color}},
            title={"text": title, "font":{"size": 15, "color": "#fff"}},
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=220, margin=dict(l=10, r=10, t=50, b=0),
    )
    return fig


def main():
    load_dotenv()
    st.set_page_config(
        page_title="Cybersecurity RAG with HF Reranking & ChatGPT Comparison",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    if "rag_pipeline" not in st.session_state:
        st.session_state["rag_pipeline"] = None
    if "knowledge_loaded" not in st.session_state:
        st.session_state["knowledge_loaded"] = False


    st.title("Cybersecurity RAG (Retriever, Reranker, ChatGPT Comparison)")
    cols = st.columns([2.6, 1])


    with cols[0]:
        user_query = st.text_input("Ask a cybersecurity question:", "")
        if st.button("Initialize RAG with Hugging Face Tools"):
            loader = LocalDocLoader()
            docs = loader.load()
            rag_pipeline = RAGPipeline()
            rag_pipeline.load_knowledge_base(docs)
            st.session_state["rag_pipeline"] = rag_pipeline
            st.session_state["knowledge_loaded"] = True
            st.success(f"Knowledge base loaded with {len(docs)} chunks.")


        if st.session_state["knowledge_loaded"] and user_query:
            cols2 = st.columns([1, 1])


            if cols2[0].button("Ask with RAG + Rerank"):
                rag = st.session_state["rag_pipeline"]
                context_docs = rag.retrieve_and_rerank(user_query, top_k_retrieve=10, top_k_rerank=3)
                rag_answer = rag.generate_qa(user_query, context_docs)
                context_text = "\n\n".join(context_docs)
                rag_tok_overlap = compute_token_overlap(context_text, rag_answer)
                rag_bigram_f1 = compute_bigram_f1(context_text, rag_answer)
                rag_sent_attr = compute_sentence_attribution(context_text, rag_answer)
                st.session_state["last_rag"] = {
                    "answer": rag_answer, "context": context_text,
                    "metrics": (rag_tok_overlap, rag_bigram_f1, rag_sent_attr)
                }
                st.markdown("#### RAG + Rerank Answer")
                st.markdown(rag_answer)
                mcols = st.columns(3)
                mcols[0].plotly_chart(build_gauge_chart(rag_tok_overlap, "Token-Overlap %", "#27b6fc"), use_container_width=True)
                mcols[1].plotly_chart(build_gauge_chart(rag_bigram_f1, "Bigram F1", "#a2fc5a"), use_container_width=True)
                mcols[2].plotly_chart(build_gauge_chart(rag_sent_attr, "Sent Attribution", "#fb4373"), use_container_width=True)
                st.markdown("##### Top Supporting Chunks/Retrieved Contexts")
                for i, doc in enumerate(context_docs, 1):
                    with st.expander(f"Context {i}"):
                        st.write(doc)


            if cols2[1].button("Ask ChatGPT Only"):
                rag = st.session_state["rag_pipeline"]
                chatgpt_answer = rag.generate_qa(user_query, context_docs=[])
                prev_context = st.session_state.get("last_rag", {}).get("context", "")
                chatgpt_tok_overlap = compute_token_overlap(prev_context, chatgpt_answer)
                chatgpt_bigram_f1 = compute_bigram_f1(prev_context, chatgpt_answer)
                chatgpt_sent_attr = compute_sentence_attribution(prev_context, chatgpt_answer)
                st.session_state["last_chatgpt"] = {
                    "answer": chatgpt_answer, "context": prev_context,
                    "metrics": (chatgpt_tok_overlap, chatgpt_bigram_f1, chatgpt_sent_attr)
                }
                st.markdown("#### ChatGPT Only Answer")
                st.markdown(chatgpt_answer)
                mcols = st.columns(3)
                mcols[0].plotly_chart(build_gauge_chart(chatgpt_tok_overlap, "Token-Overlap %", "#27b6fc"), use_container_width=True)
                mcols[1].plotly_chart(build_gauge_chart(chatgpt_bigram_f1, "Bigram F1", "#a2fc5a"), use_container_width=True)
                mcols[2].plotly_chart(build_gauge_chart(chatgpt_sent_attr, "Sent Attribution", "#fb4373"), use_container_width=True)


            if st.button("Show Last Comparison") and "last_rag" in st.session_state and "last_chatgpt" in st.session_state:
                rag_metrics = st.session_state["last_rag"]["metrics"]
                chatgpt_metrics = st.session_state["last_chatgpt"]["metrics"]
                st.markdown("### Retrieval Metrics: RAG + Rerank vs ChatGPT Only")
                st.table({
                    "Metric": ["Token-Overlap %", "Bigram F1", "Sentence Attribution"],
                    "RAG + Rerank": [f"{rag_metrics[0]:.1f}%", f"{rag_metrics[1]:.1f}%", f"{rag_metrics[2]:.1f}%"],
                    "ChatGPT Only": [f"{chatgpt_metrics[0]:.1f}%", f"{chatgpt_metrics[1]:.1f}%", f"{chatgpt_metrics[2]:.1f}%"]
                })


    with cols[1]:
        st.info("Initialize and query to begin. Uses Hugging Face retrievers/rerankers and OpenAI. Compare RAG vs ChatGPT for any query.\n\nIf you see API errors, ensure you have openai-python >=1.0.0 and the new `client.chat.completions.create()` syntax.")


if __name__ == "__main__":
    main()
