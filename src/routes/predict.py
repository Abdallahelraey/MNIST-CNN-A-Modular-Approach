from src.models.model_operations import train_model, evaluate_model, load_model, predict
from src.data.data_preprocessing import preprocess_predicted_image
from fastapi import FastAPI, File, UploadFile, APIRouter
from pydantic import BaseModel
import numpy as np
import cv2
import sys
import os
import random
import string


# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


predict_router = APIRouter()

class ImageData(BaseModel):
    image: list

@predict_router.get("/")
async def read_root():
    return {"message": "Welcome to the MNIST CNN API"}

@predict_router.post("/predict")
async def predict_digit(file: UploadFile = File(...)):

    if not file:
        return {"message": "No Upload file sent"}
    else:
        try:
            image = await file.read()
            random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            upload_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'uploaded')
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, f"image_{random_string}.jpg")
            with open(file_path, "wb") as f:
                f.write(image)
            processed_image = preprocess_predicted_image(file_path)
            predictions = predict(processed_image)
            predicted_digit = int(predictions[1])
        except FileNotFoundError as e:
            return {"error": f"[Errno 2] No such file or directory: '{file_path}'"}
        except Exception as e:
            return {"error": str(e)}
    
    return {"Predicted": predicted_digit}
