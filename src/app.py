import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

try:
    from rag_pipeline import PatchContextRAG
except Exception as import_err:
    import streamlit as st
    st.error(f"❌ Failed to import RAG pipeline: {import_err}")
    st.exception(import_err)
    st.stop()

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PatchContext - FastAPI Design RAG (Gemini)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium dark styling with CSS
st.markdown("""
<style>
    /* Main container styling */
    .reportview-container {
        background: #0f1115;
    }
    
    /* Sleek card styling for citations and metadata */
    .citation-card {
        background-color: #1e222b;
        border-left: 4px solid #10b981;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease-in-out;
    }
    
    .citation-card:hover {
        transform: translateY(-2px);
        border-left-color: #34d399;
    }
    
    .citation-title {
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 5px;
    }
    
    .citation-meta {
        font-size: 0.85em;
        color: #94a3b8;
    }
    
    .citation-link {
        color: #3b82f6;
        text-decoration: none;
        font-weight: 500;
    }
    
    .citation-link:hover {
        color: #60a5fa;
        text-decoration: underline;
    }
    
    /* Hallucination guard badge styles */
    .badge-pass {
        background-color: #064e3b;
        color: #34d399;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.9em;
        border: 1px solid #059669;
        display: inline-block;
    }
    
    .badge-warn {
        background-color: #78350f;
        color: #fbbf24;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.9em;
        border: 1px solid #d97706;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize RAG Pipeline
@st.cache_resource
def get_rag_pipeline():
    try:
        return PatchContextRAG()
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {e}")
        st.exception(e)  # Show full traceback on page so we can debug
        return None

# Sidebar Layout
with st.sidebar:
    st.image("https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png", width=120)
    st.title("PatchContext Settings")
    st.markdown("---")
    
    # Environment status checks
    gemini_key_set = bool(os.getenv("GEMINI_API_KEY"))
    github_token_set = bool(os.getenv("GITHUB_TOKEN"))
    vector_store_exists = os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "faiss_index"))
    
    st.subheader("System Status")
    if gemini_key_set:
        st.success("Gemini API Key: Configured")
    else:
        st.error("Gemini API Key: MISSING")
        st.info("Please set `GEMINI_API_KEY` in your `.env` file.")
        
    if github_token_set:
        st.success("GitHub API Token: Configured")
    else:
        st.warning("GitHub API Token: Not Configured (Rate limits may apply)")
        
    if vector_store_exists:
        st.success("Vector Database: Ready")
    else:
        st.error("Vector Database: NOT FOUND")
        st.info("Run `python src/fetch_data.py` followed by `python src/ingest.py` to build the database.")

    st.markdown("---")
    
    # Retrieve configuration parameters
    st.subheader("Retrieval Settings")
    k_docs = st.slider("Documents to Retrieve (k)", min_value=1, max_value=10, value=5)
    st.markdown("""
    **Maximal Marginal Relevance (MMR)** is used for retrieval. 
    It balances relevance with similarity diversity to ensure the LLM gets a rich set of commits, issues, and PR comments.
    """)

    st.markdown("---")
    st.markdown("Developed with ❤️ using Streamlit, LangChain, and FAISS.")

# Get RAG only if key is configured
rag = None
if gemini_key_set and vector_store_exists:
    rag = get_rag_pipeline()

# Main Application Layout
st.title("🔍 PatchContext: FastAPI Design Decision RAG")
st.markdown("""
Ask questions about **why FastAPI was designed the way it is** based on actual commit history, PR descriptions, and issue discussions.
""")

# Example Questions Panel
st.markdown("### 💡 Example Queries to try:")
cols = st.columns(3)
example_queries = [
    "Why was APIRouter created and how does it organize routes?",
    "Why was async support introduced for route endpoints?",
    "What issues led to security dependency injection refactoring?"
]

selected_query = None
for idx, q in enumerate(example_queries):
    if cols[idx].button(q, key=f"btn_{idx}"):
        selected_query = q

# Query Input Search bar
query_input = st.text_input(
    "Ask a design question about FastAPI:",
    value=selected_query or "",
    placeholder="e.g., Why was the class-based security dependency refactored?",
    key="query_bar"
)

if query_input:
    if not gemini_key_set:
        st.error("Cannot query: Gemini API Key is missing. Please resolve in the sidebar.")
    elif not vector_store_exists:
        st.error("Cannot query: Vector database is missing. Ingest data first.")
    elif rag is None:
        st.error("RAG pipeline failed to initialize. Check logs.")
    else:
        with st.spinner("Retrieving relevant developer context and generating answer..."):
            try:
                results = rag.answer_query(query_input)
                
                # Render answer and verification
                st.markdown("---")
                
                # Title and Guard Status header
                col_left, col_right = st.columns([4, 1])
                col_left.subheader("💬 Answer")
                
                guard = results.get("guard_status", {})
                if guard.get("passed", False):
                    col_right.markdown('<div class="badge-pass">🛡️ Hallucination Guard: PASSED</div>', unsafe_allow_html=True)
                else:
                    col_right.markdown('<div class="badge-warn">⚠️ Hallucination Guard: FLAGGED</div>', unsafe_allow_html=True)
                
                # Display Answer
                st.markdown(results["answer"])
                
                # NLI assessment report in expander
                with st.expander("🔍 Hallucination Guard & NLI Assessment"):
                    st.write(f"**NLI Entailment Verdict:** {guard.get('nli_verdict', 'UNKNOWN')}")
                    if guard.get("fabricated_citations"):
                        st.warning(f"**Fabricated Citations Blocked/Corrected:** {guard.get('fabricated_citations')}")
                    else:
                        st.success("No fabricated citations detected.")
                    st.markdown("**NLI Statement-by-Statement Audit:**")
                    st.text(guard.get("nli_assessment", "No assessment available."))
                
                st.markdown("---")
                
                # Citations Display
                st.subheader("📚 Verified Citations & References")
                citations = results.get("citations", [])
                
                if not citations:
                    st.info("No explicit citations found in this response. The answer may be based on general knowledge or general repository layout.")
                else:
                    cit_cols = st.columns(3)
                    for c_idx, citation in enumerate(citations):
                        col_pos = c_idx % 3
                        
                        label = citation.get("label", "[Unknown]")
                        url = citation.get("url", "")
                        src_type = citation.get("source_type", "unknown").upper()
                        src_id = citation.get("source_id", "")
                        
                        card_html = f"""
                        <div class="citation-card">
                            <div class="citation-title">{label}</div>
                            <div class="citation-meta">
                                <strong>Type:</strong> {src_type}<br/>
                                <strong>ID:</strong> {src_id}<br/>
                                <a class="citation-link" href="{url}" target="_blank">🔗 View on GitHub</a>
                            </div>
                        </div>
                        """
                        cit_cols[col_pos].markdown(card_html, unsafe_allow_html=True)
                
                # Retrieved Documents Expander
                with st.expander("📄 Raw Retrieved Context Chunks (MMR Search)"):
                    for idx_doc, doc in enumerate(results.get("retrieved_docs", [])):
                        st.markdown(f"**Chunk {idx_doc+1} - {doc.metadata.get('citation', '')} (Similarity Rank)**")
                        st.code(doc.page_content, language="markdown")
                        st.markdown("---")
                        
            except Exception as ex:
                st.error(f"Failed to generate answer: {ex}")
                st.exception(ex)
