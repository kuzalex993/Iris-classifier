# SID DDA
import uvicorn
import sklearn
from fastapi import FastAPI

app = FastAPI(
    title="Iris API",
    description="classifying plants"
)


@app.get('/')
async def index():
    return {"Message": "This is Index"}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

