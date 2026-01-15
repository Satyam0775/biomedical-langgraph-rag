import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="Biomedical RAG System",
    layout="wide"
)

# ---------- HEADER ----------
st.markdown(
    """
    <h1 style='text-align: center;'>🧬 Biomedical RAG System</h1>
    <p style='text-align: center; color: gray;'>
        Semantic vs Agentic Retrieval using LangGraph & Pinecone
    </p>
    <br>
    """,
    unsafe_allow_html=True
)

API_URL = "http://127.0.0.1:8000/query"

# ---------- CENTERED QUERY BOX ----------
left, center, right = st.columns([1, 2, 1])

with center:
    query = st.text_input(
        "Enter biomedical query",
        placeholder="e.g. deep learning models for lung cancer detection",
        label_visibility="collapsed"
    )

    search_btn = st.button("🔍 Search", use_container_width=True)

# ---------- SEARCH ----------
if search_btn and query:
    with st.spinner("Searching Pinecone via LangGraph..."):
        try:
            res = requests.post(API_URL, params={"query": query})

            if res.status_code == 200:
                data = res.json()

                st.markdown("---")

                # ---------- RESULTS ----------
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🔍 Semantic Results")
                    for r in data["semantic_results"]:
                        st.markdown(
                            f"""
                            <div style="padding:10px; border-radius:8px;
                                        border:1px solid #2c2c2c; margin-bottom:8px;">
                                <b>Agent:</b> {r['metadata']['agent_type']} <br>
                                <b>Doc ID:</b> {r['metadata']['source_doc_id']}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                with col2:
                    st.subheader("🤖 Agentic Results")
                    for r in data["agentic_results"]:
                        st.markdown(
                            f"""
                            <div style="padding:10px; border-radius:8px;
                                        border:1px solid #2c2c2c; margin-bottom:8px;">
                                <b>Agent:</b> {r['metadata']['agent_type']} <br>
                                <b>Doc ID:</b> {r['metadata']['source_doc_id']}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            else:
                st.error(f"API Error: {res.status_code}")
                st.text(res.text)

        except Exception as e:
            st.error("Failed to connect to FastAPI")
            st.exception(e)
