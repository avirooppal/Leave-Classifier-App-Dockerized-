from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from keras import layers
from tensorflow import keras

app = FastAPI()

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

tfsm_layer = tf.keras.layers.TFSMLayer('../models/2', call_endpoint='serving_default')

MODEL = tf.keras.Sequential([tfsm_layer])

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)
    print(predictions)  # Keep this for debugging
    
    # Assuming predictions is a dictionary with a key that contains the actual prediction array
    # The key name may vary, so let's get the first value in the dictionary
    prediction_array = list(predictions.values())[0]
    
    # Convert to numpy array if it's not already
    prediction_array = np.array(prediction_array)
    
    # Now we can process it as before
    if prediction_array.ndim == 2:
        predicted_class = CLASS_NAMES[np.argmax(prediction_array[0])]
        confidence = float(np.max(prediction_array[0]))
    elif prediction_array.ndim == 1:
        predicted_class = CLASS_NAMES[np.argmax(prediction_array)]
        confidence = float(np.max(prediction_array))
    else:
        raise ValueError(f"Unexpected predictions shape: {prediction_array.shape}")
    
    return {
        'class': predicted_class,
        'confidence': confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)