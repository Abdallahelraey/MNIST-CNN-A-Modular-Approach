import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape
import yaml
import os
import numpy as np
from src.data.dataset_operations import make_dataset

# def load_config():
#     with open('config/config.yaml', 'r') as file:
#         return yaml.safe_load(file)

def create_model():
    model = Sequential([
        Reshape((28, 28, 1), input_shape=(784,)),  # Add this layer to reshape the input
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model

def train_model(training_dataset):
    
    model = create_model()
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
        )

    
    history = model.fit(
        training_dataset,
        epochs=5
        )
    
    
    return model, history

def evaluate_model(model, testing_dataset):
    loss, accuracy = model.evaluate(testing_dataset)
    return{"Test Accuracy", accuracy * 100 }



def save_model(model):
    # Define the path for saving the model
    model_save_path = os.path.join("..", "models", "saved_models", "mnist_cnn_model.keras")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Save the model
    model.save(model_save_path)

def load_model():
    
    # Define the path for saving the model
    model_saved_path = os.path.join("..", "models", "saved_models", "mnist_cnn_model.keras")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_saved_path), exist_ok=True)
    # Save the model
    model = tf.keras.models.load_model(model_saved_path)
    
    return model


def predict(preprocessed_image):
    model = load_model()
    # Make prediction
    predictions = model.predict(preprocessed_image)
    predicted_digit = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    return predictions, predicted_digit, confidence
