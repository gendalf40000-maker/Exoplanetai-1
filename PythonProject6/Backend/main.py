
import os
import uuid
import shutil
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import aiohttp
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "uploads")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(
    title="FGongun Exoplanet Backend",
    description="ML API for exoplanet classification and NASA data integration",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[""],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
TRAIN_JOBS: Dict[str, Dict] = {}
class PredictInput(BaseModel):
    orbital_period: float = Field(..., gt=0, description="Orbital period in days")
    planet_radius: float = Field(..., gt=0, description="Planet radius in Earth radii")
    transit_duration: float = Field(..., gt=0, description="Transit duration in hours")

    class Config:
        schema_extra = {
            "example": {
                "orbital_period": 365.25,
                "planet_radius": 1.0,
                "transit_duration": 13.0
            }
        }


class TrainResponse(BaseModel):
    job_id: str
    status: str
    message: str


class ModelInfo(BaseModel):
    name: str
    created_at: str
    size_mb: float


class PredictionResult(BaseModel):
    model: str
    prediction: int
    probabilities: Optional[List[float]] = None
    confidence: Optional[float] = None


def model_path(name: str) -> str:
    return os.path.join(MODELS_DIR, name)


def save_model(pipeline: Pipeline, name: str) -> str:
    path = model_path(name)
    joblib.dump(pipeline, path)
    logger.info(f"Model saved: {path}")
    return path


def load_model(name: str) -> Pipeline:
    path = model_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model {name} not found")
    return joblib.load(path)


def list_models_detailed() -> List[ModelInfo]:
    models = []
    for f in os.listdir(MODELS_DIR):
        if f.endswith((".pkl", ".joblib")):
            path = model_path(f)
            created = datetime.fromtimestamp(os.path.getctime(path))
            size_mb = os.path.getsize(path) / (1024 * 1024)
            models.append(ModelInfo(
                name=f,
                created_at=created.isoformat(),
                size_mb=round(size_mb, 2)
            ))
    return sorted(models, key=lambda x: x.created_at, reverse=True)


def validate_csv_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    return all(col in df.columns for col in required_columns)


async def download_file(url: str, save_path: str) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(save_path, 'wb') as f:
                        while True:
                            chunk = await response.content.read(1024)
                            if not chunk:
                                break
                            f.write(chunk)
                    return True
        return False
    except Exception as e:
        logger.error(f"Download error: {e}")
        return False


@app.get("/", summary="Health Check")
async def root():
    return {
        "message": "FGongun Exoplanet Backend is running",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/models", response_model=List[ModelInfo], summary="List Available Models")
async def get_models():
    """Возвращает список всех доступных моделей с метаинформацией"""
    return list_models_detailed()


@app.delete("/models/{model_name}", summary="Delete Model")
async def delete_model(model_name: str):
    """Удаляет модель по имени"""
    try:
        if not model_name.endswith(('.pkl', '.joblib')):
            raise HTTPException(status_code=400, detail="Invalid model file type")

        path = model_path(model_name)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Model not found")

        os.remove(path)
        logger.info(f"Model deleted: {model_name}")
        return {"message": f"Model {model_name} deleted successfully"}

    except Exception as e:
        logger.error(f"Delete model error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete model")


@app.post("/predict", response_model=PredictionResult, summary="Predict from JSON")
async def predict_json(payload: PredictInput):
    available_models = list_models_detailed()
    if not available_models:
        raise HTTPException(status_code=400, detail="No models available. Train a model first.")

    model_file = available_models[0].name
    model = load_model(model_file)
    X = np.array([[payload.orbital_period, payload.planet_radius, payload.transit_duration]])

    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X).tolist()[0]
            pred = int(np.argmax(probs))
            confidence = max(probs)
            return PredictionResult(
                model=model_file,
                prediction=pred,
                probabilities=probs,
                confidence=round(confidence, 3)
            )
        else:
            pred = model.predict(X).tolist()[0]
            return PredictionResult(model=model_file, prediction=int(pred))

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict-file", summary="Predict from CSV File")
async def predict_file(file: UploadFile = File(..., description="CSV file with features")):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    save_name = f"predict_{uuid.uuid4().hex}_{file.filename}"
    path = os.path.join(DATA_DIR, save_name)

    try:
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        df = pd.read_csv(path)
        required_columns = ["orbital_period", "planet_radius", "transit_duration"]

        if not validate_csv_columns(df, required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain columns: {required_columns}"
            )
        available_models = list_models_detailed()
        if not available_models:
            raise HTTPException(status_code=400, detail="No models available. Train a model first.")

        model = load_model(available_models[0].name)
        X = df[required_columns].values

        results = []
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            predictions = probabilities.argmax(axis=1)

            for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
                results.append({
                    "row": i,
                    "prediction": int(pred),
                    "probabilities": probs.tolist(),
                    "confidence": round(max(probs), 3)
                })
        else:
            predictions = model.predict(X)
            for i, pred in enumerate(predictions):
                results.append({
                    "row": i,
                    "prediction": int(pred)
                })

        return {
            "model": available_models[0].name,
            "total_predictions": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"File prediction error: {e}")
        raise HTTPException(status_code=500, detail="File processing failed")
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.post("/train", response_model=TrainResponse, summary="Train New Model")
async def train_model(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(..., description="Training data CSV"),
        label_column: str = Form(..., description="Name of the target column"),
        test_size: float = Form(0.2, ge=0.1, le=0.5, description="Test set size ratio"),
        model_name: Optional[str] = Form(None, description="Custom model name")
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    job_id = uuid.uuid4().hex
    save_name = f"train_{job_id}_{file.filename}"
    path = os.path.join(DATA_DIR, save_name)

    try:
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        TRAIN_JOBS[job_id] = {
            "status": "pending",
            "message": "Queued for training",
            "model": None,
            "started_at": datetime.now().isoformat(),
            "progress": 0
        }
        background_tasks.add_task(
            _train_background,
            job_id, path, label_column, test_size, model_name
        )

        return TrainResponse(
            job_id=job_id,
            status="queued",
            message="Training job started successfully"
        )

    except Exception as e:
        logger.error(f"Training setup error: {e}")
        if os.path.exists(path):
            os.remove(path)
        raise HTTPException(status_code=500, detail="Failed to start training")


@app.get("/train/status/{job_id}", summary="Check Training Status")
async def train_status(job_id: str):
    """Возвращает статус задачи тренировки"""
    job = TRAIN_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job


@app.get("/nasa/proxy", summary="NASA Exoplanet Archive Proxy")
async def nasa_proxy(
        query: str = Form(..., description="TAP query for NASA API"),
        max_rows: int = Form(100, ge=1, le=1000, description="Maximum rows to return")
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query parameter is required")
    if any(keyword in query.upper() for keyword in ['DROP', 'DELETE', 'INSERT', 'UPDATE']):
        raise HTTPException(status_code=400, detail="Invalid query detected")
    formatted_query = f"{query} LIMIT {max_rows}"
    api_url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={formatted_query}&format=json"

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(api_url) as response:
                if response.status != 200:
                    logger.error(f"NASA API error: {response.status}")
                    raise HTTPException(
                        status_code=502,
                        detail=f"NASA API returned error: {response.status}"
                    )

                data = await response.json()
                return {
                    "query": formatted_query,
                    "row_count": len(data),
                    "data": data
                }

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="NASA API timeout")
    except Exception as e:
        logger.error(f"NASA proxy error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch NASA data")

async def _train_background(
        job_id: str,
        csv_path: str,
        label_column: str,
        test_size: float,
        model_name: Optional[str] = None
):
    try:
        TRAIN_JOBS[job_id].update({
            "status": "running",
            "message": "Loading and preprocessing data",
            "progress": 10
        })
        df = pd.read_csv(csv_path)
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in data")
        numeric_features = [
            col for col in df.columns
            if col != label_column and np.issubdtype(df[col].dtype, np.number)
        ]

        if not numeric_features:
            raise ValueError("No numeric features found for training")

        TRAIN_JOBS[job_id].update({
            "message": f"Using features: {numeric_features}",
            "progress": 20
        })
        X = df[numeric_features].fillna(0).values
        y = df[label_column].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        TRAIN_JOBS[job_id].update({
            "message": "Training Random Forest model",
            "progress": 40
        })
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ))
        ])

        pipeline.fit(X_train, y_train)

        TRAIN_JOBS[job_id].update({
            "message": "Evaluating model performance",
            "progress": 80
        })

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        final_model_name = model_name or f"model_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        save_model(pipeline, final_model_name)
        TRAIN_JOBS[job_id].update({
            "status": "completed",
            "message": f"Training completed successfully. Accuracy: {accuracy:.3f}",
            "progress": 100,
            "model": final_model_name,
            "accuracy": round(accuracy, 3),
            "classification_report": report,
            "completed_at": datetime.now().isoformat()
        })

        logger.info(f"Training completed for job {job_id} with accuracy: {accuracy:.3f}")

    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(f"Training error for job {job_id}: {e}")
        TRAIN_JOBS[job_id].update({
            "status": "error",
            "message": error_msg,
            "completed_at": datetime.now().isoformat()
        })
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error handler: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )