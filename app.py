from fastapi import FastAPI, Request
from typing import Optional
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

import warnings
warnings.filterwarnings('ignore')

app = FastAPI()


templates = Jinja2Templates(directory='templates')


origins = ["*"]

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
        

@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.get("/")
async def predictGetRouteClient(request: Request):
    try:

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"context": "Rendering"},
        )

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    
@app.get("/predict")
async def predictGetRouteClient(request: Request):
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
            "prediction.html",
            {"request": request, "context": True, "prediction": prediction[0]}
        )

    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host = APP_HOST, port =APP_PORT)

    