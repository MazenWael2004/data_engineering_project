from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import pandas as pd
import numpy as np
import os
import uuid
import time
import joblib

from typing import Optional
from pydantic import BaseModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    SGDClassifier,
)

from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

from sklearn.svm import LinearSVC

from sklearn.cluster import KMeans, MiniBatchKMeans

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)

# We will use SMOTE for imbalanced datasets if available.
try:
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False




MAX_ROWS_FOR_SMOTE = 20000 # SMOTE can be very slow on large datasets, so we set a threshold to skip it if the dataset is too big.
MAX_ROWS_FOR_RANDOM_FOREST = 100000 # Random Forest can be very slow on large datasets, so we set a threshold to skip it if the dataset is too big.
HIGH_CARDINALITY_THRESHOLD = 100 # If a categorical column has more than this many unique values, we will drop it to avoid creating too many features during one-hot encoding.

UPLOAD_DIR = "./uploads"
MODEL_DIR = "./models"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

sessions = {}



app = FastAPI(title="Fast AutoML API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




class TrainRequest(BaseModel):
    session_id: str
    task: str
    target_column: Optional[str] = None




def normalize(col): # Normalize column names by lowercasing and removing underscores and dashes
    return col.lower().replace("_", "").replace("-", "")


def get_feature_importance(model, feature_names):
    try:

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)

            if importances.ndim > 1:
                importances = importances.mean(axis=0)

        else:
            return None

        return dict(zip(feature_names, importances.tolist()))

    except Exception:
        return None


def print_training_time(model_name, start_time):
    elapsed = time.time() - start_time
    print(f"{model_name} training time: {elapsed:.2f} seconds")



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    session_id = str(uuid.uuid4())

    file_ext = file.filename.split(".")[-1].lower()

    if file_ext not in ["csv", "xlsx"]:
        raise HTTPException(
            status_code=400,
            detail="Only CSV and XLSX files are supported",
        )

    file_path = os.path.join(
        UPLOAD_DIR,
        f"{session_id}.{file_ext}"
    )

    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    sessions[session_id] = {
        "data_path": file_path
    }

    return {
        "message": "File uploaded successfully",
        "session_id": session_id,
    }




@app.post("/train")
async def train_model(request: TrainRequest):

    session_id = request.session_id
    task = request.task.lower()
    target_column = request.target_column

    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        )

    data_path = sessions[session_id]["data_path"]

   
    try:

        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path)

        elif data_path.endswith(".xlsx"):
            df = pd.read_excel(data_path)

        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type",
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading dataset: {str(e)}"
        )


    if task in ["classification", "regression"]:

        if not target_column:
            raise HTTPException(
                status_code=400,
                detail="Target column required",
            )

        if target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="Target column not found",
            )

  
    X = (
        df.drop(columns=[target_column])
        if target_column
        else df.copy()
    )

    y = (
        df[target_column]
        if target_column
        else None
    )

 
    dropped_id_columns = []

    id_like_cols = [
        col
        for col in X.columns
        if normalize(col).endswith("id")
    ]

    for col in id_like_cols:

        uniqueness_ratio = X[col].nunique() / len(X)

        if uniqueness_ratio > 0.9:
            dropped_id_columns.append(col)

    if dropped_id_columns:
        X = X.drop(columns=dropped_id_columns)

   

    if task in ["classification", "regression"]:

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if task == "classification" else None,
        )

    else:

        X_train_raw, X_test_raw = train_test_split(
            X,
            test_size=0.2,
            random_state=42,
        )

        y_train = None
        y_test = None

    

    numeric_features = X_train_raw.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train_raw.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

   

    high_cardinality_cols = [
        col
        for col in categorical_features
        if X_train_raw[col].nunique() > HIGH_CARDINALITY_THRESHOLD
    ]

    if high_cardinality_cols:

        X_train_raw = X_train_raw.drop(
            columns=high_cardinality_cols
        )

        X_test_raw = X_test_raw.drop(
            columns=high_cardinality_cols
        )

        categorical_features = [
            col
            for col in categorical_features
            if col not in high_cardinality_cols
        ]

    

    numeric_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median") # Use median for numeric imputation to be more robust to outliers
        ),
        (
            "scaler",
            StandardScaler() # z-score normalization
        ),
    ])

    categorical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore", 
                max_categories=50, # Limit the number of categories to prevent explosion of features, and drop the rest as "Other".
                sparse_output=True, 
            )
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "num",
            numeric_transformer,
            numeric_features,
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features,
        ),
    ])

    

    print("Preprocessing data...")

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    feature_names = preprocessor.get_feature_names_out().tolist()

    

    if (
        task == "classification"
        and SMOTE_AVAILABLE
        and len(X_train_raw) < MAX_ROWS_FOR_SMOTE
    ):

        print("Applying SMOTE...")

        smote = SMOTE(random_state=42)

        X_train, y_train = smote.fit_resample(
            X_train,
            y_train,
        )

   

    if task == "classification":

        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=500,
                n_jobs=-1,
            ),

            "Linear SVM": LinearSVC(), # LinearSVC is often faster than SVC with linear kernel on large datasets.

            "Random Forest": RandomForestClassifier(
                n_estimators=30,
                max_depth=10,
                n_jobs=-1,
                random_state=42,
            ),
        }

        # Remove RF on very large datasets
        if len(df) > MAX_ROWS_FOR_RANDOM_FOREST:
            models.pop("Random Forest")

        results = {}

        for name, model in models.items():

            print(f"Training {name}...")

            start = time.time()

            model.fit(X_train, y_train)

            print_training_time(name, start)

            preds = model.predict(X_test)

            results[name] = {
                "accuracy": float(
                    accuracy_score(y_test, preds)
                ),
                "precision": float(
                    precision_score(
                        y_test,
                        preds,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        y_test,
                        preds,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "f1": float(
                    f1_score(
                        y_test,
                        preds,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "confusion_matrix": confusion_matrix(
                    y_test,
                    preds,
                ).tolist(),
            }

        best_model_name = max(
            results,
            key=lambda x: results[x]["f1"]
        )

        best_model = models[best_model_name]
        best_metrics = results[best_model_name]

   
    elif task == "regression":

        models = {
            "Linear Regression": LinearRegression(
                n_jobs=-1
            ),

            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=30,
                max_depth=10,
                n_jobs=-1,
                random_state=42,
            ),
        }

        if len(df) > MAX_ROWS_FOR_RANDOM_FOREST:
            models.pop("Random Forest Regressor")

        results = {}

        for name, model in models.items():

            print(f"Training {name}...")

            start = time.time()

            model.fit(X_train, y_train)

            print_training_time(name, start)

            preds = model.predict(X_test)

            results[name] = {
                "mae": float(
                    mean_absolute_error(y_test, preds)
                ),
                "mse": float(
                    mean_squared_error(y_test, preds)
                ),
                "r2": float(
                    r2_score(y_test, preds)
                ),
            }

        best_model_name = max(
            results,
            key=lambda x: results[x]["r2"]
        )

        best_model = models[best_model_name]
        best_metrics = results[best_model_name]

    
    elif task == "clustering":

        dataset_size = len(df)

        if dataset_size > 50000:

            models = {
                "MiniBatchKMeans": MiniBatchKMeans( # MiniBatchKMeans is much faster than KMeans on large datasets, but may be less accurate.
                    n_clusters=3,
                    random_state=42,
                    batch_size=1024,
                )
            }

        else:

            models = {
                "KMeans": KMeans(
                    n_clusters=3,
                    random_state=42,
                    n_init="auto",
                ),
                "Agglomerative Clustering": AgglomerativeClustering(
                    n_clusters=3,       
                ),
            }

        results = {}

        for name, model in models.items():

            print(f"Training {name}...")

            start = time.time()

            labels = model.fit_predict(X_train)

            print_training_time(name, start)

            score = silhouette_score(
                X_train,
                labels,
                sample_size=min(5000, len(labels)),
                random_state=42,
            )

            results[name] = {
                "silhouette_score": float(score)
            }

        best_model_name = max(
            results,
            key=lambda x: results[x]["silhouette_score"]
        )

        best_model = models[best_model_name]
        best_metrics = results[best_model_name]

    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid task",
        )

  

    model_package = {
        "preprocessor": preprocessor,
        "model": best_model,
        "task": task,
        "target_column": target_column,
        "feature_names": feature_names,
    }

    model_path = os.path.join(
        MODEL_DIR,
        f"{session_id}_pipeline.pkl"
    )

    joblib.dump(model_package, model_path)

    sessions[session_id]["model_path"] = model_path



    return {
        "message": "Model trained successfully",
        "best_model": best_model_name,
        "metrics": best_metrics,
        "model_path": model_path,
        "dataset_shape": {
            "rows": len(df),
            "columns": len(df.columns),
        },
        "dropped_id_columns": dropped_id_columns,
        "high_cardinality_columns": high_cardinality_cols,
        "feature_importance": get_feature_importance(
            best_model,
            feature_names,
        ),
    }




@app.get("/download-model/{session_id}")
async def download_model(session_id: str):

    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        )

    model_path = sessions[session_id].get("model_path")

    if not model_path:
        raise HTTPException(
            status_code=404,
            detail="Model not found",
        )

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail="Model file missing",
        )

    return FileResponse(
        path=model_path,
        filename=f"automl_model_{session_id}.pkl",
        media_type="application/octet-stream",
    )




@app.get("/")
async def root():
    return {
        "message": "Fast AutoML API is running"
    }