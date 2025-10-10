import joblib
import numpy as np

# Load trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_sentiment(texts):
    """Predict sentiment for a list of texts."""
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    for i, text in enumerate(texts):
        print(f"\n📝 Text: {text}")
        print(f"→ Predicted Sentiment: {preds[i]}")
        print(f"→ Probabilities:")
        for label, p in zip(model.classes_, probs[i]):
            print(f"   {label}: {p:.3f}")

if __name__ == "__main__":
    # Example test cases
    samples = [
        "Today was… mixed, I guess. Classes went fine — I actually did well on my presentation in English, which surprised me because I barely slept last night. Everyone laughed at one of my slides, and for a moment I felt kind of proud, like maybe I’m not as invisible as I usually think I am. Lunch was okay, too. I sat with Mia and Josh again. They were joking around, and I laughed, but it felt a bit forced. I don’t know — sometimes it’s like my body is there but my head’s somewhere else. They probably didn’t notice though. I’m good at pretending. After school I just went straight home. Mom asked how my day was, and I said “fine.” It’s easier than trying to explain that weird mix of tired and empty that doesn’t make sense even to me. I ended up lying in bed scrolling for hours, feeling kind of useless for not studying when I know I should. Still, I’m trying to look at the bright side — at least I didn’t skip class today like I almost did. And I did that presentation. That has to count for something, right? Maybe tomorrow will be better. Or at least quieter."
    ]

    predict_sentiment(samples)
