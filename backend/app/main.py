from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import numpy as np
import io
import os
import uuid
import joblib
import json
from typing import Optional
from pydantic import BaseModel

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

# Regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Clustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Imbalanced
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

# In-memory session store
sessions = {}

UPLOAD_DIR = "/tmp/automl_uploads"
MODEL_DIR = "/tmp/automl_models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


class TrainRequest(BaseModel):
    session_id: str
    task: str  # classification, regression, clustering
    target_column: Optional[str] = None


@app.post("/train")
async def train_model(request: TrainRequest):
    session_id = request.session_id
    task = request.task
    target_column = request.target_column

    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Session not found")

    # ----------------------------
    # Load dataset
    # ----------------------------
    data_path = sessions[session_id]["data_path"]

    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif data_path.endswith(".xlsx"):
        df = pd.read_excel(data_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # ----------------------------
    # Split features/target
    # ----------------------------
    if task in ["classification", "regression"] and not target_column:
        raise HTTPException(status_code=400, detail="Target column required")

    X = df.drop(columns=[target_column]) if target_column else df
    y = df[target_column] if target_column else None

    # ----------------------------
    # Train/Test Split (IMPORTANT FIX)
    # ----------------------------
    if task in ["classification", "regression"]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y if task == "classification" else None
        )
    else:
        X_train, X_test = train_test_split(
            X,
            test_size=0.2,
            random_state=42
        )
        y_train = y_test = None

    # ----------------------------
    # Column detection
    # ----------------------------
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # ----------------------------
    # Preprocessing pipeline (FIT ONLY ON TRAIN)
    # ----------------------------
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Fit ONLY on training data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # ----------------------------
    # Handle imbalance (ONLY training set)
    # ----------------------------
    if task == "classification" and SMOTE_AVAILABLE:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # ----------------------------
    # Model training
    # ----------------------------
    if task == "classification":

        models = {
            "rf": RandomForestClassifier(random_state=42),
            "gb": GradientBoostingClassifier(random_state=42)
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            results[name] = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, average="weighted"),
                "recall": recall_score(y_test, preds, average="weighted"),
                "f1": f1_score(y_test, preds, average="weighted"),
                "confusion_matrix": confusion_matrix(y_test, preds).tolist()
            }

        best_model_name = max(results, key=lambda x: results[x]["f1"])
        best_model = models[best_model_name]
        best_metrics = results[best_model_name]

    elif task == "regression":

        models = {
            "rf": RandomForestRegressor(random_state=42),
            "gb": GradientBoostingRegressor(random_state=42)
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
            "kmeans": KMeans(n_clusters=3, random_state=42),
            "gmm": GaussianMixture(n_components=3, random_state=42)
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train)

            if name == "kmeans":
                labels = model.labels_
            else:
                labels = model.predict(X_train)

            score = silhouette_score(X_train, labels)

            results[name] = {
                "silhouette_score": score
            }

        best_model_name = max(results, key=lambda x: results[x]["silhouette_score"])
        best_model = models[best_model_name]
        best_metrics = results[best_model_name]

    else:
        raise HTTPException(status_code=400, detail="Invalid task")

    # ----------------------------
    # Save FULL pipeline (IMPORTANT FIX)
    # ----------------------------
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
        "model_path": model_path
    }


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




