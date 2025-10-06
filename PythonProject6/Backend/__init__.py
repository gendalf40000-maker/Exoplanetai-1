# Backend/__init__.py
import os
import logging
from pathlib import Path
__version__ = "1.0.0"
__author__ = "FGongun Team"
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "uploads"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("exoplanet-api")
from .main import app

__all__ = [
    "app",
    "logger",
    "BASE_DIR",
    "MODELS_DIR",
    "DATA_DIR"
]