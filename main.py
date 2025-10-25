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

class EnhancedTextInput(BaseModel):
    text: str
    use_ai_enhancement: bool = True

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

def get_ai_sentiment_analysis(text: str):
    """Get sentiment analysis from Gemini AI that can handle figurative language"""
    try:
        gen_model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""Analyze the sentiment of this text, paying special attention to figurative language, sarcasm, irony, and context: "{text}"

Please respond with:
1. Sentiment: [positive/negative/neutral]
2. Confidence: [0-100]
3. Reasoning: Brief explanation of why, especially noting any figurative language detected

Format your response exactly like this:
Sentiment: [sentiment]
Confidence: [number]
Reasoning: [explanation]"""
        
        response = gen_model.generate_content(prompt)
        ai_analysis = response.text
        
        # Parse AI response
        lines = ai_analysis.split('\n')
        ai_sentiment = "neutral"
        ai_confidence = 50
        ai_reasoning = "No reasoning provided"
        
        for line in lines:
            if line.startswith("Sentiment:"):
                ai_sentiment = line.split(":", 1)[1].strip().lower()
            elif line.startswith("Confidence:"):
                try:
                    ai_confidence = int(line.split(":", 1)[1].strip())
                except:
                    ai_confidence = 50
            elif line.startswith("Reasoning:"):
                ai_reasoning = line.split(":", 1)[1].strip()
        
        return ai_sentiment, ai_confidence, ai_reasoning
    except Exception as e:
        return "neutral", 0, f"AI analysis failed: {str(e)}"

def get_ai_sentiment_and_reflection(text: str):
    """Get both sentiment analysis and empathetic reflection in one prompt"""
    try:
        gen_model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""Analyze this text for sentiment, paying special attention to figurative language, sarcasm, irony, and context: "{text}"

Please respond with:
1. Sentiment: [positive/negative/neutral]
2. Confidence: [0-100] 
3. Reasoning: Brief explanation noting any figurative language detected
4. Reflection: A brief, empathetic response (2-3 sentences) that acknowledges the person's feelings, offers gentle validation/encouragement, uses a warm friendly tone, and avoids mentioning sentiment analysis directly. Speak as if talking to a friend.

Format your response exactly like this:
Sentiment: [sentiment]
Confidence: [number]
Reasoning: [explanation]
Reflection: [empathetic response]"""
        
        response = gen_model.generate_content(prompt)
        ai_analysis = response.text
        
        # Parse AI response
        lines = ai_analysis.split('\n')
        ai_sentiment = "neutral"
        ai_confidence = 50
        ai_reasoning = "No reasoning provided"
        thought = "Could not generate reflection"
        
        for line in lines:
            if line.startswith("Sentiment:"):
                ai_sentiment = line.split(":", 1)[1].strip().lower()
            elif line.startswith("Confidence:"):
                try:
                    ai_confidence = int(line.split(":", 1)[1].strip())
                except:
                    ai_confidence = 50
            elif line.startswith("Reasoning:"):
                ai_reasoning = line.split(":", 1)[1].strip()
            elif line.startswith("Reflection:"):
                thought = line.split(":", 1)[1].strip()
        
        return ai_sentiment, ai_confidence, ai_reasoning, thought
    except Exception as e:
        return "neutral", 0, f"AI analysis failed: {str(e)}", f"Could not generate reflection: {str(e)}"

def combine_sentiments(ml_sentiment, ai_sentiment, ai_confidence):
    """Combine ML and AI sentiment predictions based on AI confidence"""
    if ai_confidence < 30:
        # Low AI confidence, trust ML model more
        return ml_sentiment, "ml_primary"
    elif ai_confidence > 70:
        # High AI confidence, trust AI more
        return ai_sentiment, "ai_primary"
    else:
        # Medium confidence, use hybrid approach
        if ml_sentiment == ai_sentiment:
            return ml_sentiment, "consensus"
        else:
            # When they disagree, lean towards AI for figurative language
            return ai_sentiment, "ai_fallback"

@app.post("/predict-enhanced")
def predict_sentiment_enhanced(input: EnhancedTextInput):
    text = input.text
    
    # Get ML model prediction
    X = vectorizer.transform([text])
    ml_sentiment = model.predict(X)[0]
    
    if input.use_ai_enhancement:
        # Get AI sentiment analysis and reflection in one call
        ai_sentiment, ai_confidence, ai_reasoning, thought = get_ai_sentiment_and_reflection(text)
        
        # Combine predictions
        final_sentiment, decision_method = combine_sentiments(ml_sentiment, ai_sentiment, ai_confidence)
        
        return {
            "sentiment": final_sentiment,
            "thought": thought,
            "analysis_details": {
                "ml_sentiment": ml_sentiment,
                "ai_sentiment": ai_sentiment,
                "ai_confidence": ai_confidence,
                "ai_reasoning": ai_reasoning,
                "decision_method": decision_method
            }
        }
    else:
        # Fall back to original method
        return predict_sentiment(TextInput(text=text))
