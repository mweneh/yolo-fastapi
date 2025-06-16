# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import sys
import os

# Changed import from utils to inference_utils
from inference_utils import load_yolov5_model, predict_objects # Removed draw_boxes_on_image as it's not used in main.py

# Initialize FastAPI app
app = FastAPI(
    title="Car Damage Object Detection API (YOLOv5)",
    description="API for detecting specific car damage types and locating them with bounding boxes.",
    version="1.0.0",
)

# --- API Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    """Load the YOLOv5 model when the FastAPI application starts up."""
    try:
        load_yolov5_model()
        print("YOLOv5 model loaded successfully on startup.")
    except Exception as e:
        print(f"Failed to load YOLOv5 model on startup: {e}")
        # Make sure this exit is handled gracefully in your deployment environment
        # For local testing, sys.exit(1) is fine.
        sys.exit(1)

@app.get("/")
async def root():
    """Root endpoint for basic API health check."""
    return {"message": "Car Damage Object Detection API (YOLOv5) is running! Go to /docs for API documentation."}

@app.post("/predict/")
async def predict_damage_api(file: UploadFile = File(...)):
    """
    Performs object detection on a car image to identify and locate damage.
    Expects an image file (JPEG, PNG).
    Returns a list of detected objects (bounding boxes, class, confidence).
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')

        detections = predict_objects(img.copy())

        return JSONResponse(content={"detections": detections}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")