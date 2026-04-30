import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Automated Machine Learning App", layout="wide")

st.title("Automated Machine Learning App")

API_URL = "http://127.0.0.1:8000"

uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

task = st.selectbox("Select Task", ["classification", "regression", "clustering"])

target_column = st.text_input("Target Column (leave empty for clustering)")


if uploaded_file is not None:

    if st.button("Upload File"):

        with st.spinner("Uploading..."):

            files = {
                "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
            }

            response = requests.post(
                f"{API_URL}/upload",
                files=files
            )

            if response.status_code == 200:

                session_id = response.json()["session_id"]
                st.session_state["session_id"] = session_id

                st.success("File uploaded successfully!")

            
                import pandas as pd

                uploaded_file.seek(0)

                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.subheader("Data Preview")

                st.dataframe(df.head(10))

            else:
                st.error(response.text)


if "session_id" in st.session_state:



    if task in ["classification", "regression"] and not target_column:
        st.warning("⚠️ Please enter a target column")

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

            data = response.json()

            st.success("Model trained successfully!")

            st.subheader("Best Model")
            st.write(data["best_model"])

            st.subheader("Metrics")

            metrics = data["metrics"]

            if task == "classification":
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", round(metrics["accuracy"], 3))
                col2.metric("Precision", round(metrics["precision"], 3))
                col3.metric("Recall", round(metrics["recall"], 3))
                col4.metric("F1 Score", round(metrics["f1"], 3))


                sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(plt)


            elif task == "regression":
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", round(metrics["mae"], 3))
                col2.metric("MSE", round(metrics["mse"], 3))
                col3.metric("R2", round(metrics["r2"], 3))

            elif task == "clustering":
                st.metric("Silhouette Score", round(metrics["silhouette_score"], 3))

            

        else:
            st.error(response.text)