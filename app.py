from fastapi import FastAPI, UploadFile, File
from fastapi.responses import RedirectResponse
import uvicorn
import shutil
from src.cnnClassifier.pipeline.prediction import PredictionPipeline

app = FastAPI()
predictor = PredictionPipeline()

@app.get("/")
def home():
    return RedirectResponse(url="/docs")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predictor.predict(file_path)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
