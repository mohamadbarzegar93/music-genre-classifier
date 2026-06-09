from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.predictor import predict_genre
from api.database import init_db, log_prediction, SessionLocal, Prediction



app = FastAPI()
init_db()
@app.get("/")
def read_root():
    return {"message": "Welcome to the Music Genre Prediction API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        #Save the uploaded file to a temporary location
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        #Predict the genre using the predictor function
        result = predict_genre(temp_file_path)
        #Log the prediction to the database
        log_prediction(file.filename, result["genre"], result["confidence"])
        return JSONResponse(content = result)
    except Exception as e:
        return JSONResponse(content = {"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/history")
def history():
    session = SessionLocal()
    try:
        prediction = session.query(Prediction)\
            .order_by(Prediction.timestamp.desc())\
            .limit(10)\
            .all()
        return [
            {
                "filename": p.filename,
                "genre": p.predicted_genre,
                "confidence": p.confidence,
                "timestamp": str(p.timestamp)
            }
            for p in prediction
        ]
    finally:
        session.close()


        
    
