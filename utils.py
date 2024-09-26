import os
import pandas as pd
import requests

from dotenv import load_dotenv

load_dotenv()  

def process_reviews(filepath):
    file_extension = filepath.split('.')[-1]

    if file_extension == 'csv':
        df = pd.read_csv(filepath)
    elif file_extension == 'xlsx':
        df = pd.read_excel(filepath)
    else:
        raise Exception("Invalid file format")

    if 'review' not in df.columns:
        raise Exception("Missing 'review' column")

    reviews = df['review'].tolist()

    # Perform sentiment analysis using Groq API
    sentiment_scores = get_sentiment_analysis(reviews)
    return sentiment_scores

def get_sentiment_analysis(reviews):
    api_url = "https://api.groq.com/analyze"
    api_key = os.getenv("GROQ_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}

    positive, negative, neutral = 0, 0, 0

    for review in reviews:
        payload = {"text": review}
        response = requests.post(api_url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            positive += data['positive']
            negative += data['negative']
            neutral += data['neutral']

    total = len(reviews)
    return {
        "positive": round(positive / total, 2),
        "negative": round(negative / total, 2),
        "neutral": round(neutral / total, 2)
    }
