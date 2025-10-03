
import os
import sys

# Resolve absolute path to project root by going up two levels from this file
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_current_dir, "..", ".."))

# Append project root (which contains 'src') to sys.path if not already present
if _project_root not in sys.path:
    sys.path.append(_project_root)

import streamlit as st
from tabular_chat_rag.config import SETTINGS
from tabular_chat_rag.ingestion.loaders import load_file
from tabular_chat_rag.ingestion.preview import head_preview, schema_summary
from tabular_chat_rag.agents.sql_agent import plan_sql, run_sql
from tabular_chat_rag.agents.df_agent import plan_code, run_code
from tabular_chat_rag.rag.indexer import RowChunkIndex
from tabular_chat_rag.rag.retriever import answer_with_rag

st.set_page_config(page_title="Tabular Chat (DF | SQL | RAG)", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []
if "tables" not in st.session_state:
    st.session_state.tables = {}
if "meta" not in st.session_state:
    st.session_state.meta = {}
if "rag_index" not in st.session_state:
    st.session_state.rag_index = None

st.sidebar.title("Settings")
pipeline = st.sidebar.radio("Pipeline", ["df", "sql", "rag"], index=["df","sql","rag"].index(SETTINGS.default_pipeline))
model = st.sidebar.selectbox("Ollama model", ["phi3", "phi3:mini", "phi3:medium"], index=0)
st.sidebar.info("Ensure `ollama serve` is running and the selected model is pulled.")

st.title("Chat with CSV/XLSX using DF / SQL / RAG")

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
if uploaded:
    tables, meta = load_file(uploaded)
    st.session_state.tables = tables
    st.session_state.meta = meta

    st.subheader("Schema")
    for line in schema_summary(tables):
        st.code(line)

    st.subheader("Preview")
    for name, df in tables.items():
        st.caption(f"{name}")
        st.dataframe(head_preview(df, SETTINGS.max_preview_rows), use_container_width=True)

    if pipeline == "rag":
        idx = RowChunkIndex(SETTINGS.embed_model)
        idx.build(tables, chunk_rows=SETTINGS.chunk_rows)
        st.session_state.rag_index = idx
        st.success("RAG index built.")

# Render history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about your data…")
if question and st.session_state.tables:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("assistant"):
        if pipeline == "sql":
            try:
                sql = plan_sql(question, st.session_state.tables, model=model)
                with st.expander("Planned SQL", expanded=False):
                    st.code(sql, language="sql")
                df, _ = run_sql(sql, st.session_state.tables)
                st.dataframe(df, use_container_width=True)
                reply = f"Returned {len(df)} row(s)."
            except Exception as e:
                st.error(str(e))
                reply = f"Error: {e}"
        elif pipeline == "df":
            try:
                code = plan_code(question, st.session_state.tables, model=model)
                with st.expander("Generated pandas code", expanded=False):
                    st.code(code, language="python")
                result = run_code(code, st.session_state.tables)
                if "result_df" in result:
                    st.dataframe(result["result_df"], use_container_width=True)
                    reply = f"Returned DataFrame with {len(result['result_df'])} row(s)."
                elif "result_value" in result:
                    st.write(result["result_value"])
                    reply = f"Result: {result['result_value']}"
                else:
                    reply = "No result produced."
            except Exception as e:
                st.error(str(e))
                reply = f"Error: {e}"
        else:
            if st.session_state.rag_index is None:
                st.warning("Upload a file to build the RAG index first.")
                reply = "RAG index not available."
            else:
                answer = answer_with_rag(question, st.session_state.rag_index, SETTINGS.retrieval_k, model=model)
                st.markdown(answer)
                reply = answer
        st.session_state.history.append({"role": "assistant", "content": reply})
