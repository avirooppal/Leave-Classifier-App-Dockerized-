from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# Define the FastAPI app
app = FastAPI()

# CORS settings
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

# Load TensorFlow Serving model layer
tfsm_layer = tf.keras.layers.TFSMLayer('model/2', call_endpoint='serving_default')
MODEL = tf.keras.Sequential([tfsm_layer])

# Define class names
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Function to read and convert uploaded file to an image array
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)  # Expand dimensions to match model input

    predictions = MODEL.predict(img_batch)
    print(predictions)  # Keep this for debugging

    # Process predictions
    prediction_array = list(predictions.values())[0]  # Get the first value in the dictionary

    prediction_array = np.array(prediction_array)  # Convert to numpy array if it's not already

    if prediction_array.ndim == 2:
        predicted_class = CLASS_NAMES[np.argmax(prediction_array[0])]
        confidence = float(np.max(prediction_array[0]))
    elif prediction_array.ndim == 1:
        predicted_class = CLASS_NAMES[np.argmax(prediction_array)]
        confidence = float(np.max(prediction_array))
    else:
        raise ValueError(f"Unexpected predictions shape: {prediction_array.shape}")

    print({
        'class': predicted_class,
        'confidence': confidence
    })
    return {
        'class': predicted_class,
        'confidence': confidence
    }

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
