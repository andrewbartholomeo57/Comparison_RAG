import streamlit as st
import os
import tempfile
import torch
import pandas as pd

from pipeline import process_documents_parallel
from vectordb import load_embedding_model
from diff_engine import match_chunks, generate_structural_differences, detect_change_types
from ollama_engine import explain_differences, parse_llm_response


st.set_page_config(page_title="Document Diff Analyzer", layout="wide")

st.title("OLLAMA + RAG Document Difference Analyzer")
st.markdown("Upload two document versions and analyze their differences.")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    st.success(f"GPU Detected: **{gpu_name}** - Running in GPU-accelerated mode")
elif torch.backends.mps.is_available():
    st.success("Apple MPS GPU Detected - Running in GPU-accelerated mode")
else:
    st.warning("No GPU detected - Running in CPU mode (slower)")

col1, col2 = st.columns(2)
with col1:
    file_v1 = st.file_uploader("Upload Document 1 (e.g. v1 / original)", type=["pdf", "docx", "pptx"])
with col2:
    file_v2 = st.file_uploader("Upload Document 2 (e.g. Final / updated)", type=["pdf", "docx", "pptx"])

interpret_prompt = st.text_area(
    "Prompt to interpret document content",
    placeholder="Explain the document and tell me which parts cannot be interpreted (charts, images, etc.)"
)

diff_prompt = st.text_area(
    "Prompt to analyze differences",
    value="Analyze the key differences between the two document versions."
)

if st.button("Analyze Documents"):

    if not file_v1 or not file_v2:
        st.error("Please upload both documents.")
    else:
        with st.spinner("Loading embedding model..."):
            # Pre-load the model on the main thread before parallel processing starts.
            embedding_model = load_embedding_model()

        with st.spinner("Processing both documents in parallel..."):

            temp_dir = tempfile.mkdtemp()
            path_v1 = os.path.join(temp_dir, file_v1.name)
            path_v2 = os.path.join(temp_dir, file_v2.name)

            with open(path_v1, "wb") as f:
                f.write(file_v1.read())
            with open(path_v2, "wb") as f:
                f.write(file_v2.read())

            vectordb_v1, limitations_v1, vectordb_v2, limitations_v2 = \
                process_documents_parallel(path_v1, path_v2)

            matches = match_chunks(vectordb_v1, vectordb_v2, embedding_model)
            diff_results = generate_structural_differences(matches)
            change_types = detect_change_types(matches, vectordb_v1, vectordb_v2)

            limitation_report = {"v1": limitations_v1, "v2": limitations_v2}

        with st.spinner("Analyzing with OLLAMA..."):
            raw_explanation = explain_differences(
                diff_results,
                change_types,
                diff_prompt,
                limitation_report
            )
            summary, table_rows = parse_llm_response(raw_explanation)

        st.success("Analysis Complete!")

        st.subheader("Change Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Modified", len(change_types["modified"]))
        c2.metric("Added", len(change_types["added"]))
        c3.metric("Removed", len(change_types["removed"]))
        c4.metric("Total Differences", len(table_rows))

        st.divider()

        st.subheader("Summary")
        st.info(summary)

        st.divider()

        st.subheader("Key Differences Comparison Table")

        if table_rows:
            df = pd.DataFrame(table_rows, columns=["Aspect", "Document 1", "Document 2"])
            df.index = df.index + 1

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=False,
                column_config={
                    "Aspect": st.column_config.TextColumn("Aspect", width="medium"),
                    "Document 1": st.column_config.TextColumn(f"{file_v1.name}", width="large"),
                    "Document 2": st.column_config.TextColumn(f"{file_v2.name}", width="large"),
                }
            )

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Table as CSV",
                data=csv,
                file_name="key_differences.csv",
                mime="text/csv"
            )
        else:
            st.warning("No structured table returned. Check debug expander below.")

        st.divider()

        with st.expander("Interpretation Limitations"):
            st.json(limitation_report)

        with st.expander("Raw LLM Response (Debug)"):
            st.text(raw_explanation)

        with st.expander("Structural Diffs (Unified Diff)"):
            for i, diff in enumerate(diff_results[:10]):
                st.markdown(f"**Pair {i+1} - Similarity: `{diff['similarity']:.3f}`**")
                st.code(diff["diff"], language="diff")

        with st.expander(f"Added - only in Document 2 ({len(change_types['added'])} chunks)"):
            for i, text in enumerate(change_types["added"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.text(text[:500] + ("..." if len(text) > 500 else ""))

        with st.expander(f"Removed - only in Document 1 ({len(change_types['removed'])} chunks)"):
            for i, text in enumerate(change_types["removed"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.text(text[:500] + ("..." if len(text) > 500 else ""))
