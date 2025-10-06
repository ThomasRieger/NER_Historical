import streamlit as st
st.set_page_config(page_title="Historical Text Analyzer", layout="wide")
import pandas as pd
import os
from Website.utils_old import load_ner_model
from user_page import run_user_page

run_user_page()
