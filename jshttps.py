from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

current_data = {}

@app.post("/update")
async def update_vitals(data: dict):
    global current_data
    current_data = data
    # We return immediately so the feeder doesn't time out
    return {"status": "success"}

@app.get("/latest")
async def get_latest():
    return current_data