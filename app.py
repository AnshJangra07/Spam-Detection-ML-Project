import os
from threading import Lock
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from uvicorn import run as app_run
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
load_dotenv()


from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.constant.application import *
from src.logger import logging

import warnings
warnings.filterwarnings('ignore')

app = FastAPI()
training_status = {
    "status": "idle",
    "message": "Train a model to see its evaluation metrics.",
    "metrics": None,
    "model_published": False,
}
training_status_lock = Lock()


templates = Jinja2Templates(directory='templates')


origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:8000"
    ).split(",")
    if origin.strip()
]

app.mount("/static", StaticFiles(directory="static"), name="static")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.text: Optional[str] = None

    async def get_text_data(self):
        form =  await self.request.form()
        self.text = form.get('input_text')
        

def run_training_task():
    try:
        result = TrainPipeline().run_pipeline()
        with training_status_lock:
            training_status.update(
                status="completed",
                message=result["message"],
                metrics=result["metrics"],
                model_published=result["model_published"],
            )
    except Exception as error:
        logging.error(f"Training failed: {error}")
        with training_status_lock:
            training_status.update(
                status="failed",
                message="Training failed. Check the application logs for details.",
                metrics=None,
                model_published=False,
            )


@app.post("/train")
async def train_route(background_tasks: BackgroundTasks):
    with training_status_lock:
        if training_status["status"] == "running":
            raise HTTPException(status_code=409, detail="Training is already in progress")
        training_status.update(
            status="running",
            message="Training...",
            metrics=None,
            model_published=False,
        )

    background_tasks.add_task(run_training_task)
    return {"status": "running", "message": "Training..."}


@app.get("/train/status")
async def train_status_route():
    with training_status_lock:
        return dict(training_status)


@app.get("/")
async def home_route(request: Request):
    try:

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"context": "Rendering"},
        )

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    
@app.get("/predict")
async def predict_form_route(request: Request):
    try:

        return templates.TemplateResponse(
            request=request,
            name="prediction.html",
            context={"context": False},
        )
        
    except Exception as e:
        return Response(f"Error Occurred! {e}")
    
@app.post("/predict")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        
        await form.get_text_data()
        
        input_data = [form.text]
        print(form.text)
        
        # return Response(f"got data is : {input_data[0]}")
    
        
        prediction_pipeline = PredictionPipeline()
        prediction: int = prediction_pipeline.run_pipeline(input_data=input_data)
        
        print(f"the prediction is : {prediction}")
       
        
        return templates.TemplateResponse(
            request=request,
            name="prediction.html",
            context={"context": True, "prediction": prediction[0]},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed") from e


if __name__ == "__main__":
    app_run(app, host = APP_HOST, port =APP_PORT)

    