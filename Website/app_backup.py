import streamlit as st
import pandas as pd

def extract_named_entities(text):
    return []

def extract_triples(text):
    return []

def calculate_similarity(triples1, triples2):
    if not triples1 or not triples2:
        return 0.0
    common = [t for t in triples1 if t in triples2]
    return len(common) / max(len(triples1), len(triples2))

st.title("NNER for Historical Text Comparison")

CSV_PATH = "./DATA/data_v1.csv"
st.info(f"Loading data from `{CSV_PATH}`")

try:
    df = pd.read_csv(CSV_PATH)

    if 'topic' not in df.columns or 'sentence' not in df.columns:
        st.error("CSV must have 'topic' and 'sentence' columns.")
    else:
        st.write("### Preview of Loaded Data")
        st.dataframe(df.head())

        topics = df['topic'].dropna().unique().tolist()

        st.header("Select Topics and Optionally Enter Custom Text")

        col1, col2 = st.columns([3, 1])
        with col1:
            topic1 = st.selectbox("Topic 1", topics, key="topic1")
        with col2:
            use_custom1 = st.checkbox("Custom Input", key="custom1")

        custom_text1 = ""
        if use_custom1:
            custom_text1 = st.text_area("Enter Custom Text for Topic 1", key="text1")

        col3, col4 = st.columns([3, 1])
        with col3:
            topic2 = st.selectbox("Topic 2", topics, index=1 if len(topics) > 1 else 0, key="topic2")
        with col4:
            use_custom2 = st.checkbox("Custom Input", key="custom2")

        custom_text2 = ""
        if use_custom2:
            custom_text2 = st.text_area("Enter Custom Text for Topic 2", key="text2")

        # Get actual sentences
        sentence1 = custom_text1 if use_custom1 else df[df['topic'] == topic1]['sentence'].values[0]
        sentence2 = custom_text2 if use_custom2 else df[df['topic'] == topic2]['sentence'].values[0]

        st.markdown("Final Texts for Comparison")
        st.write(f"**{topic1} Text:** {sentence1}")
        st.write(f"**{topic2} Text:** {sentence2}")

        st.markdown("---")
        st.header("Named Entity Recognition (NER)")

        if st.button("Extract Entities"):
            st.subheader(f"Entities in Topic 1 ({topic1})")
            st.write(extract_named_entities(sentence1))

            st.subheader(f"Entities in Topic 2 ({topic2})")
            st.write(extract_named_entities(sentence2))

        st.markdown("---")
        st.header("Triple Comparison")

        if st.button("Compare Triples"):
            triples1 = extract_triples(sentence1)
            triples2 = extract_triples(sentence2)

            st.subheader(f"Triples in {topic1}")
            st.write(triples1)

            st.subheader(f"Triples in {topic2}")
            st.write(triples2)

except FileNotFoundError:
    st.error(f"CSV file not found at `{CSV_PATH}`. Please check the path.")
