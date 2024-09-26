from fastapi import FastAPI, File, UploadFile, HTTPException
import os
from utils import process_reviews
import shutil

# Initialize FastAPI application
app = FastAPI()

# Define the upload folder path and allowed file extensions
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# Helper function to validate file type
def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    
    Args:
    filename (str): Name of the uploaded file
    
    Returns:
    bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Endpoint to analyze sentiment from uploaded file.
    
    Args:
    file (UploadFile): The uploaded file containing reviews
    
    Returns:
    dict: Sentiment analysis results
    
    Raises:
    HTTPException: If file type is not supported or if an error occurs during processing
    """
    # Check for allowed file type
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Save the uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Process the file and perform sentiment analysis
        sentiment_scores = process_reviews(filepath)
        return sentiment_scores
    except Exception as e:
        # If an error occurs during processing, raise an HTTPException
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up: remove the uploaded file after processing
        os.remove(filepath)

if __name__ == "__main__":
    # Create the upload folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Run the FastAPI application using uvicorn server
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)