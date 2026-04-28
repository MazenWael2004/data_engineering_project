import streamlit as st
import requests

st.title("AutoML Platform")

API_URL = "http://127.0.0.1:8000"

uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

task = st.selectbox("Select ML Task", ["classification", "regression", "clustering"])

target_column = st.text_input("Target Column (leave empty for clustering)")

session_id = None

# =========================
# STEP 1: UPLOAD FILE
# =========================
if uploaded_file is not None and st.button("Upload File"):
    with st.spinner("Uploading..."):

        files = {"file": uploaded_file.getvalue()}

        response = requests.post(
            f"{API_URL}/upload",
            files={"file": uploaded_file}
        )

        if response.status_code == 200:
            session_id = response.json()["session_id"]
            st.success("File uploaded successfully!")
            st.session_state["session_id"] = session_id
        else:
            st.error(response.text)

# =========================
# STEP 2: TRAIN MODEL
# =========================
if "session_id" in st.session_state:

    st.write(f"Session ID: {st.session_state['session_id']}")

    if st.button("Train Model"):

        payload = {
            "session_id": st.session_state["session_id"],
            "task": task,
            "target_column": target_column if target_column else None
        }

        with st.spinner("Training model..."):
            response = requests.post(
                f"{API_URL}/train",
                json=payload
            )

        if response.status_code == 200:
            st.success("Model trained successfully!")
            st.json(response.json())
        else:
            st.error(response.text)