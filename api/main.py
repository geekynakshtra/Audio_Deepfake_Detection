from fastapi import FastAPI, UploadFile, File
from model.detector import AudioDeepfakeDetector
import shutil
import os
import uuid

app = FastAPI()
detector = AudioDeepfakeDetector()

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4().hex}.wav"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = detector.predict(temp_filename)
    finally:
        os.remove(temp_filename)

    return result
