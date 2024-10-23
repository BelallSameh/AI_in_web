from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from fastapi.responses import StreamingResponse

# Print the current working directory
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

# Adjust the path if necessary based on your file location
file_path = os.path.join(current_dir, "app/model/model.h5")

# Check if the model file exists
if os.path.exists(file_path):
    print(f"Model found at: {file_path}")
    model = tf.keras.models.load_model(file_path)
    print("Model loaded successfully.")
else:
    print(f"Model not found at: {file_path}")
    exit()

# Initialize FastAPI app
app = FastAPI()

# Preprocess the input image (128x128)
def preprocess_image(image: Image.Image):
    # Resize the image to 128x128 as expected by the model
    img = image.resize((128, 128))  # Adjust size to 128x128
    img = np.array(img)  # Convert image to numpy array
    img = img.astype('float32') / 255.0  # Normalize to [0, 1]
    if img.ndim == 2:  # If grayscale, add a channel dimension
        img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, height, width, channels)
    return img

# Post-process the model output (for multi-class segmentation)
def postprocess_output(output):
    output = np.squeeze(output, axis=0)  # Remove batch dimension
    
    # Convert the multi-channel output to single-channel by taking argmax (class with highest probability)
    output_argmax = np.argmax(output, axis=-1)  # Shape will become (128, 128)
    
    # Normalize the output for visualization (optional)
    output_rescaled = (output_argmax * (255 / output_argmax.max())).astype(np.uint8)
    
    # Convert numpy array back to PIL image
    img = Image.fromarray(output_rescaled)
    return img

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        image = Image.open(file.file)
        
        # Preprocess the image
        input_data = preprocess_image(image)
        
        # Make a prediction
        prediction = model.predict(input_data)
        
        # Post-process the prediction
        output_image = postprocess_output(prediction)
        
        # Convert the output image to a byte stream
        img_io = io.BytesIO()
        output_image.save(img_io, format='PNG')
        img_io.seek(0)
        
        # Return the image as a streaming response
        return StreamingResponse(img_io, media_type="image/png")
    
    except Exception as e:
        return {"error": str(e)}

# Run the app if this script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
