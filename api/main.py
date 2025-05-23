from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

app = FastAPI()
MODEL = tf.keras.models.load_model('./saved_models/3')
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/ping')
async def ping():
    return "Hi! I am alive!"

def read_image_from_file(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
    
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image = read_image_from_file(await file.read())
    img_batch = np.expand_dims(image, axis=0)
    
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(float(np.max(predictions[0])), 2)
    
    return {
        "class": predicted_class,
        "confidence": confidence
    }

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)