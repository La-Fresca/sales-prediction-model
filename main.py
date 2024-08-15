from fastapi import FastAPI
import pandas as pd

df = pd.read_excel('Coffee_Shop_Sales.xlsx')

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

