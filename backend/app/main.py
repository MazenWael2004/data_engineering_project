from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import uuid
import joblib
from typing import Optional
from pydantic import BaseModel
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, silhouette_score
)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


app = FastAPI(title="AutoML API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

UPLOAD_DIR = "./uploads"
MODEL_DIR = "./models"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


class TrainRequest(BaseModel):
    session_id: str
    task: str
    target_column: Optional[str] = None


def normalize(col):
    return col.lower().replace("_", "").replace("-", "")

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())

    file_ext = file.filename.split(".")[-1].lower()
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}.{file_ext}")

    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    sessions[session_id] = {
        "data_path": file_path
    }

    return {"session_id": session_id}


@app.post("/train")
async def train_model(request: TrainRequest):
    session_id = request.session_id
    task = request.task
    target_column = request.target_column

    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Session not found")

    data_path = sessions[session_id]["data_path"]

    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif data_path.endswith(".xlsx"):
        df = pd.read_excel(data_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    if task in ["classification", "regression"] and not target_column:
        raise HTTPException(status_code=400, detail="Target column required")

    X = df.drop(columns=[target_column]) if target_column else df
    y = df[target_column] if target_column else None

    
    id_like_cols = [
        col for col in X.columns
        if normalize(col).endswith("id")
    ]

    cols_to_drop = []
    for col in id_like_cols:
        unique_ratio = X[col].nunique() / len(X)
        if unique_ratio > 0.9:
            cols_to_drop.append(col)

    X = X.drop(columns=cols_to_drop, errors="ignore")

    # we identify numeric and categorical features for preprocessing
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # preprocessing pipelines for numeric and categorical features
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), # we use median to handle outliers better than mean
        ("scaler", StandardScaler()) # we standardize numeric features to have mean=0 and std=1, which can help many models perform better
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features), 
        ("cat", categorical_transformer, categorical_features)
    ])

    # SPLIT
    if task in ["classification", "regression"]:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if task == "classification" else None
        )
    else:
        X_train_raw, X_test_raw = train_test_split(
            X,
            test_size=0.2,
            random_state=42
        )
        y_train = y_test = None

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # SMOTE
    if task == "classification" and SMOTE_AVAILABLE:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # ================= MODELS =================
    if task == "classification":
        models = {

            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Support Vector Machine": SVC(random_state=42), # Added class_weight to help with imbalance
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            results[name] = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
                "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
                "f1": f1_score(y_test, preds, average="weighted", zero_division=0),
                "confusion_matrix": confusion_matrix(y_test, preds).tolist()
            }

        best_model_name = max(results, key=lambda x: results[x]["f1"])
        best_model = models[best_model_name]
        best_metrics = results[best_model_name]

    elif task == "regression":
        models = {
            "RandomForest": RandomForestRegressor(random_state=42),
            "LinearRegression": LinearRegression(),
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            results[name] = {
                "mae": mean_absolute_error(y_test, preds),
                "mse": mean_squared_error(y_test, preds),
                "r2": r2_score(y_test, preds)
            }

        best_model_name = max(results, key=lambda x: results[x]["r2"])
        best_model = models[best_model_name]
        best_metrics = results[best_model_name]

    elif task == "clustering":
        models = {
            "KMeans": KMeans(n_clusters=3, random_state=42),
            "GMM": GaussianMixture(n_components=3, random_state=42)
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train)

            labels = model.labels_ if name == "KMeans" else model.predict(X_train)
            score = silhouette_score(X_train, labels)

            results[name] = {"silhouette_score": score}

        best_model_name = max(results, key=lambda x: results[x]["silhouette_score"])
        best_model = models[best_model_name]
        best_metrics = results[best_model_name]

    else:
        raise HTTPException(status_code=400, detail="Invalid task")

    # SAVE PIPELINE
    model_package = {
        "preprocessor": preprocessor,
        "model": best_model,
        "task": task,
        "target_column": target_column
    }

    model_path = os.path.join(MODEL_DIR, f"{session_id}_pipeline.pkl")
    joblib.dump(model_package, model_path)

    sessions[session_id]["model_path"] = model_path

    return {
        "message": "Model trained successfully",
        "best_model": best_model_name,
        "metrics": best_metrics,
        "model_path": model_path,
        "dropped_id_columns": cols_to_drop
    }