"""
TruthLayer — Streamlit UI
Run with: streamlit run app.py
"""

import streamlit as st
import time
import tempfile
import os
from itertools import combinations
from core.ollama_client import query_multiple, check_ollama_available, get_installed_models
from core.trust_score import compute_trust_score
from core.reasoning_graph import render_reasoning_graph
from rag.retriever import document_store

st.set_page_config(
    page_title="TruthLayer",
    page_icon="🛡️",
    layout="wide",
)

# ── Custom CSS for Rich UI ────────────────────────────────────
st.markdown("""
<style>
/* Modern styling for the app */
div[data-testid="stAppViewContainer"] {
    background: #0a0a0a;
    color: #f8fafc;
}
div[data-testid="stHeader"] {
    background: rgba(15, 23, 42, 0.8);
    backdrop-filter: blur(10px);
}
.stButton>button {
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}
.stMetric {
    background: rgba(255, 255, 255, 0.05);
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.title("🛡️ TruthLayer")
    st.markdown("<p style='font-size: 1.1rem; color: #94a3b8;'>Pioneering transparency and reliability: ensuring AI remains grounded and explainable.</p>", unsafe_allow_html=True)
with col2:
    allowed_models = ["truthlayer-med-q4:latest", "gemma4:e4b", "gemma4:e2b"]
    if "model" not in st.session_state or st.session_state.model not in allowed_models:
        st.session_state.model = allowed_models[0]
    model = st.selectbox("Model", allowed_models, key="model", help="The local Ollama model to use for the analysis.")
with col3:
    if "runs" not in st.session_state:
        st.session_state.runs = 3
    runs = st.number_input("Verification Runs", min_value=1, max_value=10, value=3, key="runs", help="Number of times to query the model. Higher runs increase reliability by comparing multiple responses for consistency.")
st.divider()

# ── Sidebar UI for RAG ──────────────────────────────────────────
with st.sidebar:
    st.header("📚 Document Grounding (RAG)")
    use_rag = st.checkbox("Enable RAG", value=False, help="Ground the model's answers in your uploaded documents.")
    
    st.divider()
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader("Upload Medical PDFs", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Ingesting documents..."):
                total_chunks = 0
                for file in uploaded_files:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name
                        
                    chunks_added = document_store.ingest_pdf(tmp_path, file.name)
                    total_chunks += chunks_added
                    os.unlink(tmp_path) # Cleanup
                    
                st.success(f"Successfully processed {len(uploaded_files)} files ({total_chunks} chunks stored).")
        else:
            st.warning("Please upload a file first.")

# ── State Management ────────────────────────────────────────────
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "cot_parsed" not in st.session_state:
    st.session_state.cot_parsed = None
if "elapsed" not in st.session_state:
    st.session_state.elapsed = None
# (State variables already initialized in header except query states)

# ── Input ─────────────────────────────────────────────────────
# chat_input naturally supports Shift+Enter for newline and Enter to submit
query = st.chat_input("Enter a medical query (Shift+Enter for new line)")

if query:
    st.session_state.last_query = query
    st.session_state.analysis_result = None  # Clear previous results to trigger new analysis

# ── Ollama status ─────────────────────────────────────────────
ollama_ok = check_ollama_available(model)
if not ollama_ok:
    st.error(f"⚠️ Model {model} not available. Is Ollama running?")

# ── Empty State (ChatGPT Style) ────────────────────────────────
if not st.session_state.last_query:
    st.markdown("""
    <style>
    /* Center the chat input for the empty state */
    div[data-testid="stChatInputContainer"] {
        bottom: 40vh !important;
    }
    .greeting-container {
        position: fixed;
        bottom: 50vh;
        left: 0;
        width: 100%;
        text-align: center;
        z-index: 999;
    }
    .greeting {
        font-size: 2.2rem;
        font-weight: 500;
        color: #f8fafc;
    }
    .pill-container {
        position: fixed;
        bottom: 30vh;
        left: 0;
        width: 100%;
        display: flex;
        justify-content: center;
        gap: 1rem;
        z-index: 999;
    }
    .pill {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        color: #cbd5e1;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    </style>
    <div class="greeting-container">
        <div class="greeting">What medical query can I help you with?</div>
    </div>
    <div class="pill-container">
        <div class="pill">💊 Check symptoms</div>
        <div class="pill">🏥 Treatment options</div>
        <div class="pill">🔬 Medical concepts</div>
    </div>
    """, unsafe_allow_html=True)

# ── Analysis Execution ───────────────────────────────────────
# If we have a query but no result yet, run the analysis
if st.session_state.last_query and not st.session_state.analysis_result:
    if not ollama_ok:
        st.error("Ollama is not running. Start it first.")
        st.stop()

    st.divider()
    st.markdown("### Query")
    st.info(st.session_state.last_query)
    
    with st.status("Analyzing query...", expanded=True) as status:
        context = None
        if use_rag:
            st.write("🔍 Retrieving relevant context from ChromaDB...")
            context = document_store.search(st.session_state.last_query, top_k=3)
            st.session_state.retrieved_context = context

        st.write(f"🔄 Executing {runs} verification passes using Gemma 4...")
        t0 = time.time()
        raw_responses, parsed = query_multiple(st.session_state.last_query, runs=runs, model=model, context=context)
        elapsed_time = time.time() - t0

        if not raw_responses:
            st.error("No responses received from Ollama. Try again.")
            st.session_state.last_query = None # Reset
            st.stop()

        st.write("🧮 Computing trust score...")
        res = compute_trust_score(raw_responses)
        
        # Save to state
        st.session_state.analysis_result = res
        st.session_state.cot_parsed = parsed
        st.session_state.elapsed = elapsed_time
        
        status.update(label=f"✅ Analysis complete in {elapsed_time:.1f}s", state="complete")
        st.rerun() # Force a rerun so that the results display outside the status block

# ── Display Results ──────────────────────────────────────────
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    cot_parsed = st.session_state.cot_parsed
    elapsed = st.session_state.elapsed
    
    st.divider()

    st.markdown("### Query")
    st.info(st.session_state.last_query)
    
    st.divider()

    # ── Best Answer ───────────────────────────────────────────
    best_idx = result["best_response_idx"]
    best_cot = cot_parsed[best_idx]

    st.markdown("### ✅ Best Answer")
    st.info(best_cot["answer"])

    # ── Trust Score ───────────────────────────────────────────
    tier = result["tier"]
    score = result["trust_score"]

    if tier == "HIGH":
        st.success(f"**Trust Score: {score}/100 — HIGH ✅**")
    elif tier == "MODERATE":
        st.warning(f"**Trust Score: {score}/100 — MODERATE ⚠️**")
    else:
        st.error(f"**Trust Score: {score}/100 — LOW 🔴**")

    st.caption(result["advice"])
    
    st.divider()

    # ── Retrieved Context (If RAG enabled) ──────────────────────
    if st.session_state.get("retrieved_context"):
        with st.expander("📚 Retrieved Context", expanded=True):
            for ctx in st.session_state.retrieved_context:
                st.markdown(f"```text\n{ctx}\n```")

    # ── Reasoning Graph (Visible by default) ───────────────────────────────────
    st.markdown("### 🧠 Model Reasoning Graph")
    render_reasoning_graph(result, cot_parsed)
    
    st.markdown("### 🔍 Detailed Analysis")

    # ── Signal Breakdown ──────────────────────────────────────
    with st.expander("📊 Signal Breakdown"):
        s = result["signals"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Consistency", f"{s['consistency']['score']:.0f}/100")
            st.progress(int(s['consistency']['score']))
            st.caption(s['consistency']['verdict'])
        with col2:
            st.metric("Confidence Language", f"{s['confidence_language']['score']:.0f}/100")
            st.progress(int(s['confidence_language']['score']))
            st.caption(s['confidence_language']['verdict'])
        with col3:
            st.metric("Length Variance", f"{s['length_variance']['score']:.0f}/100")
            st.progress(int(s['length_variance']['score']))

        if s["consistency"]["disagreements"]:
            for d in s["consistency"]["disagreements"]:
                st.warning(f"⚠️ {d}")

    # ── All Responses ─────────────────────────────────────────
    with st.expander("🔎 See all responses"):
        if runs == 1:
            temps = [0.7]
        else:
            temps = [round(0.1 + (0.9 * i / (runs - 1)), 2) for i in range(runs)]
            
        for i, cot in enumerate(cot_parsed):
            is_best = (i == best_idx)
            label = f"Run {i+1} · temp {temps[i] if i < len(temps) else '?'}" + (" ⭐ best" if is_best else "")
            if is_best:
                st.success(f"**{label}**\n\n{cot['answer'][:400]}")
            else:
                st.markdown(f"**{label}**\n\n{cot['answer'][:400]}")
            st.divider()

    # ── Consistency details ───────────────────────────────────
    with st.expander("📐 Consistency details"):
        st.caption(s["consistency"]["verdict"])
        # Dynamically generate pairs
        pairs = [f"Run {i+1} vs Run {j+1}" for i, j in combinations(range(runs), 2)]
        for pair, score in zip(pairs, s["consistency"]["pairwise"]):
            st.markdown(f"**{pair}**: {score*100:.1f}/100")
            st.progress(score)

    # ── Hedging language ──────────────────────────────────────
    with st.expander("🗣️ Hedging language detected"):
        hedges = s["confidence_language"]["hedges_found"]
        if hedges:
            for h in hedges:
                st.markdown(f"- `{h}`")
        else:
            st.success("No significant hedging language detected.")

    st.divider()
    st.caption(f"⏱️ {elapsed:.1f}s · {runs} runs · {model} · Running 100% locally via Ollama")