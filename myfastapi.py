# SID DDA
import uvicorn
import sklearn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

app = FastAPI(
    title="Iris API",
    description="classifying plants"
)

templates = Jinja2Templates(directory="templates")
@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse("index.html", context= {
        "request": request,
        "somevar": 2
    })


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
