from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

app = FastAPI()
endpoint = "https://potato-tf-serving.onrender.com/v1/models/potatoes_model/versions/1:predict"
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/ping')
async def ping():
    return "Hi! I am alive!"

def read_image_from_file(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)).convert('RGB'))
    return image
    
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image = read_image_from_file(await file.read())
    img_batch = np.expand_dims(image, axis=0)
    
    json_data = {
        "instances": img_batch.tolist()
    }

    print(json_data)
    
    response = requests.post(endpoint, json=json_data, timeout=10)

    print(response.text)
    if response.status_code != 200:
        return {"error": "TensorFlow Serving returned an error", "details": response.text}

    try:
        prediction = np.array(response.json()["predictions"][0])
    except KeyError:
        return {"error": "Invalid response from TensorFlow Serving", "response": response.json()}
    prediction = np.array(response.json()["predictions"][0])
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = round(float(np.max(prediction)), 2)

    return{
        "class": predicted_class,
        "confidence": confidence    
    }
    
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
