# AI-Sentiment Analysis

## Overview

AI-Sentiment Analysis is a Python-based application that processes customer reviews from CSV or Excel files. It performs sentiment analysis using a Groq API to classify reviews as Positive, Negative, or Neutral. The application outputs the counts and proportions of each sentiment category along with any unclassified reviews.

## Tech Stack used -

- Python
- FastAPI
- Groq sdk
- pandas

## Features

- Process reviews from both CSV and Excel file formats.
- Utilize a Groq API to analyze sentiments of reviews.
- Return counts and proportions of Positive, Negative, and Neutral sentiments.
- Provide a summary of unclassified reviews for further analysis.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yashbudhia/AI-sentiment-analysis.git
   cd AI-sentiment-analysis
   ```

2. **Install required packages**:
   Make sure you have Python installed. You can create a virtual environment (recommended) and install the required packages using:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Create a `.env` file in the root of the project and add your Groq API key as follows:
     ```bash
     GROQ_API_KEY=your_api_key_here
     ```

## Usage

- To process a file of reviews, run the following command in your terminal:
  ```bash
  uvicorn main:app --reload
  ```
- Then Go to Postman or any other api tester , and send a POST request to http://127.0.0.1:8000/analyze
- In the body go to form data , create a new field and add your csv or xlsx file there and send the post request
- You will get the reviews counted in postive, negative and neutral based on the sentiment.
- format of count -

```{
    "positive": 25,
    "negative": 23,
    "neutral": 3
}
```
