from fastapi import APIRouter, File, UploadFile
from brain_tumor_detection.services.inference import predict

router = APIRouter()

@router.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    try:
        return predict(file)
    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}
