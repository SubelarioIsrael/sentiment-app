from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model + vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI(
    title="Sentiment Analysis API",
    description="Predicts sentiment (positive, neutral, negative) from text input",
    version="1.0"
)

# Define request schema
class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is running!"}

@app.post("/predict")
def predict_sentiment(input: TextInput):
    text = input.text
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return {"sentiment": pred}
