import models.classifier as clf
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from joblib import load
from models.iris import Iris
import uvicorn
import jinja2
from fastapi.templating import Jinja2Templates

app = FastAPI(
    title="Iris API",
    description="classifying plants"
)

templates = Jinja2Templates(directory="templates")

@app.on_event('startup')
def load_model():
    clf.model = load('models/final_model.joblib')

@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse("index.html", context= {
        "request": request,
        "somevar": 2
    })


@app.post("/predict")
async def predicted(request: Request, SL: float = Form(...), SW: float = Form(...), PL: float = Form(...), PW: float = Form(...)):
    data = [[SL, SW, PL, PW]]
    prediction = clf.model.predict(data)
    prediction_str = str(prediction)
    return templates.TemplateResponse('index.html', context={
        'request': request,
        "prediction": prediction[0],
        "SL": SL,
        "SW": SW,
        "PL": PL,
        "PW": PW})

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)