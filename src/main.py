from fastapi import FastAPI
from src.brain_tumor_detection.api.routes import router

app = FastAPI(
    title="Brain Tumor MRI Classifier",
    description="Upload an MRI scan and get a tumor prediction using ONNX model",    version="1.0.0"
)

app.include_router(router)
