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
GDRIVE_DATA = Path(os.environ.get("GDRIVE_DATA", PROJECT / "data"))

# =========================================================
# Könyvtárak
# =========================================================
if IS_COLAB:
    DATA_DIR = Path("/content/drive/MyDrive/MetabolKD/data")
    OUTPUT_DIR = Path("/content/drive/MyDrive/MetabolKD/outputs")
else:
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
                        
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
SPLITS_DIR = INTERIM_DIR / "splits"
MODELS_DIR = INTERIM_DIR / "models"

FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOGS_DIR = OUTPUT_DIR / "logs"                   

