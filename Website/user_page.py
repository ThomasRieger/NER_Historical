import streamlit as st
import pandas as pd
import io
from utils import extract_named_entities, extract_triples, save_to_database

def run_user_page():
    st.title("User Interface - Historical Text Analysis")

    input_mode = st.radio("Choose input type", ["Manual Text", "Upload CSV"])

    results = []

    if input_mode == "Manual Text":
        text = st.text_area("Enter your text:")
        if st.button("Extract"):
            ner = extract_named_entities(text)
            triples = extract_triples(text)

            result = {"text": text, "ner": ner, "triples": triples}
            results.append(result)
            save_to_database(result)

            st.write("Named Entities:", ner)
            st.write("Triples:", triples)

            result_df = pd.DataFrame(results)

            # Create downloadable CSV (Thai-safe)
            csv_buffer = io.BytesIO()
            result_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            csv_buffer.seek(0)

            st.download_button(
                label="Download CSV",
                data=csv_buffer,
                file_name="results.csv",
                mime="text/csv"
            )

    else:
        file = st.file_uploader("Upload CSV", type=["csv"])

        if file:
            try:
                df = pd.read_csv(file)
                st.subheader("Preview of Uploaded File")
                st.dataframe(df.head())

                if 'sentence' not in df.columns:
                    st.error("❌ CSV format is incorrect. Required column: `sentence`.")
                    st.info("Please use a format like this:")
                    sample_df = pd.DataFrame({
                        "sentence": [
                            "อยุธยาเป็นเมืองหลวงเก่า"
                        ]
                    })
                    st.dataframe(sample_df)
                else:
                    if st.button("Extract from CSV"):
                        for _, row in df.iterrows():
                            text = row['sentence']
                            ner = extract_named_entities(text)
                            triples = extract_triples(text)
                            result = {"text": text, "ner": ner, "triples": triples}
                            results.append(result)
                            save_to_database(result)

                        result_df = pd.DataFrame(results)

                        st.success("Extraction complete!")
                        st.dataframe(result_df)

                        # Create downloadable CSV (Thai-safe)
                        csv_buffer = io.BytesIO()
                        result_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        csv_buffer.seek(0)

                        st.download_button(
                            label="Download CSV",
                            data=csv_buffer,
                            file_name="results.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"Error reading file: {e}")
