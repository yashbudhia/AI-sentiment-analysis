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

    review_column = next((col for col in df.columns if col.strip().lower() == 'review'), None)
    if review_column is None:
        raise Exception("Missing 'Review' column or equivalent")

    df = df.dropna(subset=[review_column])
    reviews = df[review_column].tolist()

    print(f"Total number of non-empty reviews: {len(reviews)}")

    sentiment_counts = get_sentiment_analysis(reviews)
    
    total_classified = sum(sentiment_counts.values())
    sentiment_proportions = {
        key: round(value / total_classified, 2) if total_classified > 0 else 0.0
        for key, value in sentiment_counts.items()
    }
    
    return {
        "counts": sentiment_counts,
        "proportions": sentiment_proportions,
        "total_reviews": len(reviews),
        "unclassified": len(reviews) - total_classified
    }

def get_sentiment_analysis(reviews, batch_size=10):
    overall_sentiment = {"positive": 0, "negative": 0, "neutral": 0}

    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i+batch_size]
        prompt = (
            "Analyze the sentiment of each of the following reviews. "
            "For each review, respond with ONLY 'Positive', 'Negative', or 'Neutral'. "
            "After classifying all reviews, provide a summary count in the format:\n"
            "Positive: <count>\nNegative: <count>\nNeutral: <count>\n\n"
            "Reviews:\n"
        )
        prompt += "\n".join(f"{j+1}. {review}" for j, review in enumerate(batch))

        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
            )

            response_content = chat_completion.choices[0].message.content.strip()
            print(f"Batch {i//batch_size + 1} response:\n{response_content}\n")

            lines = response_content.split('\n')
            summary_start = next(i for i, line in enumerate(lines) if line.startswith("Positive:"))
            summary = lines[summary_start:]

            batch_sentiment = {"positive": 0, "negative": 0, "neutral": 0}
            for line in summary:
                if line.startswith("Positive:"):
                    batch_sentiment["positive"] = int(line.split(":")[1].strip())
                elif line.startswith("Negative:"):
                    batch_sentiment["negative"] = int(line.split(":")[1].strip())
                elif line.startswith("Neutral:"):
                    batch_sentiment["neutral"] = int(line.split(":")[1].strip())

            for key in overall_sentiment:
                overall_sentiment[key] += batch_sentiment[key]

        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")

    print(f"Overall sentiment counts: {overall_sentiment}")
    return overall_sentiment

