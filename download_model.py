import bz2
import shutil
import urllib.request
from pathlib import Path

MODEL_URL = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "shape_predictor_68_face_landmarks.dat"
COMPRESSED_PATH = BASE_DIR / "shape_predictor_68_face_landmarks.dat.bz2"

if MODEL_PATH.exists():
    print("Dlib landmark model already exists.")
    raise SystemExit(0)

print("Downloading dlib 68 landmark model...")

urllib.request.urlretrieve(MODEL_URL, COMPRESSED_PATH)

print("Extracting model...")

with bz2.open(COMPRESSED_PATH, "rb") as source:
    with open(MODEL_PATH, "wb") as target:
        shutil.copyfileobj(source, target)

COMPRESSED_PATH.unlink(missing_ok=True)

print("Dlib landmark model ready.")