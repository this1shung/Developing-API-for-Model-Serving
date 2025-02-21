from fastapi import FastAPI, File, UploadFile, HTTPException
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import logging
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

try:
    model_path = "mnist_model.onnx"
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logger.info(f"Model loaded successfully. Input name: {input_name}, Output name: {output_name}")
except Exception as e:
    logger.error("FAILED TO LOAD THE MODEL")
    raise

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes))

        image = image.convert('L')

        image = image.resize((28,28))

        img_array = np.array(image,dtype=np.float32)

        img_array = img_array / 255.0

        img_array = img_array.reshape(1,1,28,28)

        return img_array
    
    except Exception as e:
        logger.error(f"Error preprocessing image")
        raise


@app.get('/')
def index():
    return {'message': 'MNIST ONNX Model Recognition API'}

@app.get('/health')
def health_check():
    return {
        'status': 'healthy',
        'model': 'loaded',
        'input_name': input_name,
        'output_name': output_name
    }

@app.post('/predict')
async def predict_digit(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        logger.info(f"Received prediction request")

        img_array = preprocess_image(image_bytes)

        predictions = session.run([output_name],{input_name: img_array})[0]

        predicted_class = int(np.argmax(predictions[0]))

        confidence = float(predictions[0][predicted_class])

        logger.info(f"Predicted digit :{predicted_class}, Confidence: {confidence}")

        return{
            'predicted_digit': predicted_class,
            'confidence': confidence,
            'probabilities': predictions[0].tolist()
        }
    except Exception as e:
        logger.error(f"Error during prediction")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#"uvicorn FastAPI:app --reload" to run