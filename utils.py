import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Groq client with the API key from environment variables
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def process_reviews(filepath):
    """
    Process reviews from a specified file and return sentiment analysis results.
    
    Parameters:
        filepath (str): The path to the file containing reviews.
        
    Returns:
        dict: A dictionary with sentiment counts, proportions, and total reviews.
    """
    
    # Extract the file extension to determine the file type
    file_extension = filepath.split('.')[-1]

    # Read the data from the file based on its extension
    if file_extension == 'csv':
        df = pd.read_csv(filepath)  # Read CSV file
    elif file_extension == 'xlsx':
        df = pd.read_excel(filepath)  # Read Excel file
    else:
        raise Exception("Invalid file format")  # Raise error if the file format is unsupported

    # Find the review column in a case-insensitive manner
    review_column = next((col for col in df.columns if col.strip().lower() == 'review'), None)
    if review_column is None:
        raise Exception("Missing 'Review' column or equivalent")  # Raise error if review column is missing

    # Drop rows where the review column is NaN (empty)
    df = df.dropna(subset=[review_column])
    reviews = df[review_column].tolist()  # Convert review column to a list

    # Print total number of non-empty reviews for debugging
    print(f"Total number of non-empty reviews: {len(reviews)}")

    # Get sentiment analysis results
    sentiment_counts, unclassified_reviews = get_sentiment_analysis(reviews)
    
    # Calculate proportions of each sentiment category
    total_classified = sum(sentiment_counts.values())  # Total classified reviews
    sentiment_proportions = {
        key: round(value / total_classified, 2) if total_classified > 0 else 0.0
        for key, value in sentiment_counts.items()  # Calculate proportions for each sentiment
    }
    
    return {
        "counts": sentiment_counts,  # Counts of each sentiment
        "proportions": sentiment_proportions,  # Proportions of each sentiment
        "total_reviews": len(reviews),  # Total number of reviews processed
        "unclassified": len(unclassified_reviews),  # Number of unclassified reviews
        "unclassified_reviews": unclassified_reviews  # List of unclassified reviews
    }

def get_sentiment_analysis(reviews, batch_size=10):
    """
    Analyze sentiment of a list of reviews and categorize them into positive, negative, and neutral.

    Parameters:
        reviews (list): List of reviews to analyze.
        batch_size (int): Number of reviews to process in each API call.

    Returns:
        tuple: Overall sentiment counts and a list of unclassified reviews.
    """
    overall_sentiment = {"positive": 0, "negative": 0, "neutral": 0}  # Initialize sentiment counts
    unclassified_reviews = []  # List to hold unclassified reviews

    # Process reviews in batches
    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i+batch_size]  # Create a batch of reviews
        prompt = (
            "Analyze the sentiment of each of the following reviews. "
            "For each review, respond with ONLY 'Positive', 'Negative', or 'Neutral'. "
            "After classifying all reviews, provide a summary count in the format:\n"
            "Positive: <count>\nNegative: <count>\nNeutral: <count>\n\n"
            "Reviews:\n"
        )
        prompt += "\n".join(f"{j+1}. {review}" for j, review in enumerate(batch))  # Add reviews to prompt

        try:
            # Send the prompt to the Groq API to classify sentiments
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
            )

            response_content = chat_completion.choices[0].message.content.strip()  # Get the response content
            print(f"Batch {i//batch_size + 1} response:\n{response_content}\n")  # Print response for debugging

            lines = response_content.split('\n')  # Split response into lines
            classifications = []  # List to hold classifications

            # Extract sentiment classifications from the response
            for line in lines:
                if line and line[0].isdigit():  # Check if the line starts with a number (indicating a review)
                    parts = line.split('.', 1)  # Split by the first period
                    if len(parts) > 1:
                        classification = parts[1].strip().lower()  # Get the sentiment classification
                        if classification in ['positive', 'negative', 'neutral']:
                            classifications.append(classification)  # Add valid classification to list

            # Find and parse the summary of sentiment counts from the response
            summary_start = next((i for i, line in enumerate(lines) if line.startswith("Positive:")), -1)
            if summary_start != -1:
                summary = lines[summary_start:]  # Extract the summary lines
                batch_sentiment = {"positive": 0, "negative": 0, "neutral": 0}  # Initialize batch sentiment counts
                for line in summary:
                    for key in batch_sentiment:  # Check each line for sentiment counts
                        if line.lower().startswith(key):
                            batch_sentiment[key] = int(line.split(":")[1].strip())  # Parse count

                # Update overall sentiment counts
                for key in overall_sentiment:
                    overall_sentiment[key] += batch_sentiment[key]

            # Check for unclassified reviews in this batch
            if len(classifications) < len(batch):
                unclassified_reviews.extend(batch[len(classifications):])  # Add unclassified reviews to the list

        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")  # Log error
            unclassified_reviews.extend(batch)  # Add entire batch to unclassified if there's an error

    # Final pass for unclassified reviews
    if unclassified_reviews:
        print("Processing unclassified reviews:")  # Indicate unclassified review processing
        for review in unclassified_reviews[:]:  # Iterate over a copy of the list
            prompt = f"Analyze the sentiment of the following review. Respond with ONLY 'Positive', 'Negative', or 'Neutral'.\n\nReview: {review}"
            try:
                # Send each unclassified review for classification
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-8b-8192",
                )
                response = chat_completion.choices[0].message.content.strip().lower()  # Get the response
                if response in overall_sentiment:
                    overall_sentiment[response] += 1  # Increment the appropriate sentiment count
                    unclassified_reviews.remove(review)  # Remove from unclassified list
                    print(f"Classified as {response}: {review}")  # Log classification
                else:
                    print(f"Failed to classify: {review}")  # Log failure to classify
            except Exception as e:
                print(f"Error processing review: {str(e)}")  # Log error for unclassified review

    print(f"Overall sentiment counts: {overall_sentiment}")  # Print final sentiment counts
    return overall_sentiment, unclassified_reviews  # Return overall counts and unclassified reviews
