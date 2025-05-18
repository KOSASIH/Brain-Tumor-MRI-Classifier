import os
import numpy as np
import tensorflow as tf
from models.tumor_model import create_model

def create_test_model():
    """
    Create and save a test model for the application
    """
    print("Creating a test model for TumorDetect AI...")
    
    # Create model directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create the model
    model = create_model(input_shape=(224, 224, 3), num_classes=4)
    
    # Save the model
    model_path = 'models/brain_tumor_model.h5'
    model.save(model_path)
    
    print(f"Test model created and saved to {model_path}")
    print("This is an untrained model for testing purposes only.")
    print("For a production environment, you should train the model with real data.")

if __name__ == "__main__":
    create_test_model()