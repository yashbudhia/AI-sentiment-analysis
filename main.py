from fastapi import FastAPI, File, UploadFile, HTTPException
import os
from utils import process_reviews
import shutil

app = FastAPI()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# Helper function to validate file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
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
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
