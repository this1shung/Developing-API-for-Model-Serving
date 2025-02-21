from fastapi import FastAPI, UploadFile, HTTPException, File
import tritonclient.http as httpclient
import numpy as np
from PIL import Image
import io
import logging
import uvicorn


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()


TRITON_SERVER_URL = "localhost:8000"  
MODEL_NAME = "mnist_cnn"  
INPUT_NAME = "input.1"  
OUTPUT_NAME = "19"  

try:
    client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)
    logger.info(f"✅ Connected to Triton Inference Server at {TRITON_SERVER_URL}")
except Exception as e:
    logger.error(f"❌ Failed to connect to Triton Server: {e}")
    raise SystemExit(1)  

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  
        image = image.resize((28, 28))  
        img_array = np.array(image, dtype=np.float32) / 255.0  
        img_array = img_array.reshape(1, 1, 28, 28)  
        return img_array
    except Exception as e:
        logger.error(f"⚠️ Error during image preprocessing: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image format or processing error.")

@app.get("/")
def index():
    return {"message": "MNIST Triton Model API"}
@app.get("/health")
def health_check():
    try:
        if client.is_server_live():
            return {"status": "healthy", "triton_server": "live", "model": MODEL_NAME}
        else:
            return {"status": "unhealthy", "triton_server": "down"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        logger.info(f"📥 Received prediction request for file: {file.filename}")

        img_array = preprocess_image(image_bytes)

        inputs = httpclient.InferInput(INPUT_NAME, img_array.shape, datatype="FP32")
        inputs.set_data_from_numpy(img_array, binary_data=True)

        outputs = httpclient.InferRequestedOutput(OUTPUT_NAME, binary_data=True)

        results = client.infer(model_name=MODEL_NAME, inputs=[inputs], outputs=[outputs])

        predictions = results.as_numpy(OUTPUT_NAME)[0]
        predicted_class = int(np.argmax(predictions))
        confidence = float(predictions[predicted_class])

        logger.info(f"✅ Predicted digit: {predicted_class}, Confidence: {confidence:.4f}")

        return {
            "predicted_digit": predicted_class,
            "confidence": confidence,
            "probabilities": predictions.tolist()
        }
    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":   
    uvicorn.run(app, host="127.0.0.1", port=8000)
