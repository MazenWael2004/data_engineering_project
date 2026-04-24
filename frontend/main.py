import streamlit as st

uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
task = st.selectbox("Select ML Task", ["Classification", "Regression", "Clustering"])