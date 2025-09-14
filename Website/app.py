import streamlit as st

st.set_page_config(page_title="Historical Text Analyzer", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["User Page", "Admin Page"])

if page == "User Page":
    from user_page import run_user_page
    run_user_page()

elif page == "Admin Page":
    from admin_page import run_admin_page
    run_admin_page()
