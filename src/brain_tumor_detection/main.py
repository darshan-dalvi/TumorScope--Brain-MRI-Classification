from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from brain_tumor_detection.api.routes import router

app = FastAPI(
    title="Brain Tumor MRI Classifier",
    description="Upload an MRI scan and get a tumor prediction "
    "using ONNX model",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=False,  # Must be False for wildcard origin
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
