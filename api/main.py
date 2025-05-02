from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from model.detector import AudioDeepfakeDetector
import shutil
import os
import uuid

app = FastAPI()

# Enable CORS so frontend (running separately) can access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

# Load model once at startup
detector = AudioDeepfakeDetector()

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_filename = f"temp_{uuid.uuid4().hex}.wav"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = detector.predict(temp_filename)
    finally:
        os.remove(temp_filename)

    return result


