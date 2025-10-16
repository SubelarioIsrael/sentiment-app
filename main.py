from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import google.generativeai as genai
import os

# # Only load dotenv if not running in Render
# if not os.getenv("RENDER"):
#     from dotenv import load_dotenv
#     load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

for m in genai.list_models():
    print(m.name)

# Load sentiment model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI(
    title="Sentiment Analysis API",
    description="Predicts sentiment (positive, neutral, negative) from text input and generates AI reflections.",
    version="1.1"
)

class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is running!"}

@app.post("/predict")
def predict_sentiment(input: TextInput):
    text = input.text
    X = vectorizer.transform([text])
    sentiment = model.predict(X)[0]

    # Generate Gemini reflection
    try:
        gen_model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""Based on the text: "{text}"
        
The sentiment analysis detected: {sentiment}

Please provide a brief, empathetic response (2-3 sentences) that:
- Acknowledges the person's feelings without being overly clinical
- Offers gentle validation or encouragement as appropriate
- Uses a warm, human tone as if speaking to a friend
- Avoids mentioning the sentiment analysis directly

Focus on being genuinely helpful and supportive. And if they are doing well, make sure to celebrate that! Also make sure to speak as if you're talking directly to the person like a friend."""
        response = gen_model.generate_content(prompt)
        thought = response.text
    except Exception as e:
        thought = f"Could not generate reflection: {str(e)}"

    return {
        "sentiment": sentiment,
        "thought": thought
    }
