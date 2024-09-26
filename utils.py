import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def process_reviews(filepath):
    file_extension = filepath.split('.')[-1]

    if file_extension == 'csv':
        df = pd.read_csv(filepath)
    elif file_extension == 'xlsx':
        df = pd.read_excel(filepath)
    else:
        raise Exception("Invalid file format")

    # Check for column name in a case-insensitive manner
    review_column = None
    for col in df.columns:
        if col.strip().lower() == 'review':
            review_column = col
            break

    if review_column is None:
        raise Exception("Missing 'Review' column or equivalent")

    # Remove any rows with empty reviews
    df = df.dropna(subset=[review_column])
    reviews = df[review_column].tolist()

    print(f"Total number of non-empty reviews: {len(reviews)}")

    # Perform sentiment analysis
    sentiment_counts = get_sentiment_analysis(reviews)
    
    # Calculate proportions
    total = sum(sentiment_counts.values())
    sentiment_proportions = {
        key: round(value / total, 2) if total > 0 else 0.0
        for key, value in sentiment_counts.items()
    }
    
    return {
        "counts": sentiment_counts,
        "proportions": sentiment_proportions,
        "total_reviews": len(reviews)
    }

def get_sentiment_analysis(reviews):
    prompt = (
        "Please analyze the following reviews and provide the counts of positive, negative, and neutral reviews. "
        "Ensure that every review is classified and the total count matches the number of reviews provided. "
        "Respond in the following format:\n"
        "Positive: <count>\n"
        "Negative: <count>\n"
        "Neutral: <count>\n"
        "Total: <total count>\n\n"
        "Here are the reviews:\n"
    )
    prompt += "\n".join(reviews)

    # Send the prompt to the Groq chatbot
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )

    # Debugging: Print the full response
    print(chat_completion)

    # Check if choices were returned
    if not chat_completion.choices:
        raise Exception("No choices returned from the Groq API")

    # Get the response from the chatbot
    response_content = chat_completion.choices[0].message.content.strip()

    # Initialize sentiment counts
    sentiment_scores = {
        "positive": 0,
        "negative": 0,
        "neutral": 0,
    }

    # Extract counts from the response content
    try:
        total_from_api = 0
        for line in response_content.splitlines():
            if line.startswith("Positive:"):
                sentiment_scores["positive"] = int(line.split(":")[1].strip())
            elif line.startswith("Negative:"):
                sentiment_scores["negative"] = int(line.split(":")[1].strip())
            elif line.startswith("Neutral:"):
                sentiment_scores["neutral"] = int(line.split(":")[1].strip())
            elif line.startswith("Total:"):
                total_from_api = int(line.split(":")[1].strip())
                
        # Verify that all reviews were classified
        total_classified = sum(sentiment_scores.values())
        if total_classified != len(reviews) or total_classified != total_from_api:
            print(f"Warning: Mismatch in review counts. Classified: {total_classified}, Original: {len(reviews)}, API Total: {total_from_api}")
            
    except (IndexError, ValueError) as e:
        raise Exception("Error parsing the response content: " + str(e))

    return sentiment_scores

