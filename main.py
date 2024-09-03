from fastapi import FastAPI
from src.routes import predict
import uvicorn


app = FastAPI()
app.include_router(predict.predict_router)

print("Model Pridected the image successfully.")

    
