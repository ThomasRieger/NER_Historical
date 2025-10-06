import streamlit as st
import pandas as pd
import io
from utils import extract_named_entities, extract_triples, save_to_database

def run_user_page():
    if "disclaimer_shown" not in st.session_state:
        st.session_state.disclaimer_shown = False

    if not st.session_state.disclaimer_shown:
        with st.expander("⚠️ Disclaimer", expanded=True):
            st.markdown("""
            **Please Note:**  
            This tool uses automated Named Entity Recognition (NER) and relation extraction.  
            Results may contain inaccuracies or incomplete information.  
            Always **review and verify** before using the extracted data for research, publication, or decision-making.
            """)
            if st.button("I Understand"):
                st.session_state.disclaimer_shown = True
        st.stop()

    st.title("Historical Text Analysis - NER & Triple Extraction")
    input_mode = st.radio("Select Input Type", ["Manual Text", "Upload CSV"])

    if "results" not in st.session_state:
        st.session_state.results = []

    if input_mode == "Manual Text":
        text = st.text_area("Enter your text:")
        if st.button("Extract"):
            if text.strip():
                with st.spinner("Extracting entities and triples..."):
                    ner = extract_named_entities(text)
                    triples = extract_triples(text)
                result = {
                    "text": text,
                    "ner": "; ".join([f"{t}:{l}" for t, l in ner]),
                    "triples": "; ".join([f"{s}-{r}-{o}" for s, r, o in triples])
                }
                st.session_state.results.append(result)
                save_to_database(result)

        if st.session_state.results:
            st.subheader("All Extracted Entries")
            result_df = pd.DataFrame(st.session_state.results)
            st.dataframe(result_df)

            csv_buffer = io.BytesIO()
            result_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
            csv_buffer.seek(0)
            st.download_button("Download All Results as CSV", csv_buffer, "results.csv", "text/csv")

    else:
        file = st.file_uploader("Upload your CSV file", type=["csv"])
        if file:
            try:
                df = pd.read_csv(file)
                st.subheader("Preview of Uploaded File")
                st.dataframe(df.head())

                if "sentence" not in df.columns:
                    st.error("❌ CSV must have a 'sentence' column.")
                    return

                if st.button("Extract from CSV"):
                    with st.spinner("Processing all rows..."):
                        for _, row in df.iterrows():
                            text = row["sentence"]
                            ner = extract_named_entities(text)
                            triples = extract_triples(text)
                            result = {
                                "text": text,
                                "ner": "; ".join([f"{t}:{l}" for t, l in ner]),
                                "triples": "; ".join([f"{s}-{r}-{o}" for s, r, o in triples])
                            }
                            st.session_state.results.append(result)
                            save_to_database(result)

                    result_df = pd.DataFrame(st.session_state.results)
                    st.success("✅ Extraction complete!")
                    st.dataframe(result_df)

                    csv_buffer = io.BytesIO()
                    result_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
                    csv_buffer.seek(0)
                    st.download_button("Download All Results as CSV", csv_buffer, "results.csv", "text/csv")

            except Exception as e:
                st.error(f"Error reading CSV: {e}")