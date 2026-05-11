# src/config.py
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import json
import os
import sys


# -------------------------------------------------
# Környezet felismerés és SEED beállítás
# -------------------------------------------------

IS_COLAB = "google.colab" in sys.modules
SEED = int(os.environ.get("SEED", "42"))

# -------------------------------------------------
# Projekt gyökér
# -------------------------------------------------

if IS_COLAB:
    PROJECT_ROOT = Path("/content/MetabolKD")

else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
                 
# Forrás (Drive / közös cache): ha nincs a két .txt, ide töltjük le.
GDRIVE_DATA = Path("/content/drive/MyDrive/MetabolKD/data")


# =========================================================
# Könyvtárak
# =========================================================
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
INTERIM_DIR = DATA_DIR / "interim"
SPLITS_DIR = INTERIM_DIR / "splits"
MODELS_DIR = INTERIM_DIR / "models"

FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOGS_DIR = OUTPUT_DIR / "logs"                   

