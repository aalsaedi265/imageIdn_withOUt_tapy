

import io
import pickle
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Load the model
with open('mnist_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File is not an image.")
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert('L')
        pil_image = ImageOps.invert(pil_image)
        pil_image = pil_image.resize((28, 28), Image.LANCZOS)
        img_array = np.array(pil_image).reshape(1, -1)
        
        prediction = model.predict(img_array)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
