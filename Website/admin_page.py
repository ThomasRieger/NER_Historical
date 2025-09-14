import streamlit as st
import pandas as pd

ADMIN_PASSWORD = "1q2w"

def run_admin_page():
    st.title("Admin Dashboard")

    password = st.text_input("Enter admin password", type="password")
    if password != ADMIN_PASSWORD:
        st.error("Incorrect password.")
        return

    try:
        df = pd.read_csv("data/database.csv")
        st.success("Access granted.")
        st.write("Full User Submission History")
        st.dataframe(df)
        st.download_button("Download Full Data", df.to_csv(index=False), "full_data.csv", "text/csv")
    except FileNotFoundError:
        st.warning("No user data found yet.")
