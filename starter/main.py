# Put the code for your API here.
from pydantic import BaseModel

from starter.starter.ml.data import clean_data
from starter.starter.train_model import train_save_model, make_prediction

from fastapi import FastAPI


class DataRow(BaseModel):
    rowNumber: int


# Instantiate the app.
app = FastAPI()


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


@app.post("/prediction/")
async def create_item(item: DataRow):
    prediction = make_prediction(item.rowNumber)
    return {"prediction": prediction.item()}


if __name__ == "__main__":
    clean_data()
    train_save_model()
    # make_prediction(10)
