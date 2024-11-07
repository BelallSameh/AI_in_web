from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, Column, Integer, String, event
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import os
import logging

# Database URL configuration (you can modify this if needed)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/mydatabase")

# Database connection setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Set up logging for database actions
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the SegmentationResult model first
class SegmentationResult(Base):
    __tablename__ = "segmentation_results"
    id = Column(Integer, primary_key=True, index=True)
    result = Column(String, index=True)

# Create the tables if they don't exist
Base.metadata.create_all(bind=engine)

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

# Preprocess the input image (resize and normalize)
def preprocess_image(image: Image.Image):
    img = image.resize((128, 128))  # Resize to 128x128
    img = np.array(img).astype('float32') / 255.0
    if img.ndim == 2:  # If grayscale, add a channel dimension
        img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, height, width, channels)
    return img

# Post-process the model output (for multi-class segmentation)
def postprocess_output(output):
    output = np.squeeze(output, axis=0)  # Remove batch dimension
    output_argmax = np.argmax(output, axis=-1)  # Shape will become (128, 128)
    output_rescaled = (output_argmax * (255 / output_argmax.max())).astype(np.uint8)
    return Image.fromarray(output_rescaled)

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
        
        # Save the segmentation result to the database
        db = SessionLocal()  # Open a session
        db_result = SegmentationResult(result="Segmentation completed")
        db.add(db_result)  # Add the result to the session
        db.commit()  # Commit the transaction
        db.refresh(db_result)  # Refresh to get the inserted ID

        # Log the insertion
        logger.info(f"Inserted a new record with ID {db_result.id} and result: {db_result.result}")
        db.close()  # Close the session
        
        # Return the image as a streaming response
        return StreamingResponse(img_io, media_type="image/png")
    
    except Exception as e:
        return {"error": str(e)}

# Run the app if this script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
