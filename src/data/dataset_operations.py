import tensorflow as tf
import yaml
import os
from src.data.data_preprocessing import preprocess_fn

# def load_config():
#     with open('config/config.yaml', 'r') as file:
#         return yaml.safe_load(file)

def make_dataset():
    # config = load_config()
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
    
    return train_dataset, test_dataset


def save_raw_dataset (dataset):
    # Define the path for saving the raw_data_save_path
    raw_data_save_path = os.path.join("..", "data/raw", "MNSIT_RAW") 

    # Ensure the directory exists
    os.makedirs(os.path.dirname(raw_data_save_path), exist_ok=True)

    tf.data.Dataset.save(dataset,raw_data_save_path)

def save_processed_dataset (train_dataset,test_dataset):
    # Define the path for saving the Processed_data
    Train_Processed_data_save_path = os.path.join("..", "data/Processed", "MNSIT_Train_Processed") 
    Test_Processed_data_save_path = os.path.join("..", "data/Processed", "MNSIT_Test_Processed") 

    # Ensure the directory exists
    os.makedirs(os.path.dirname(Train_Processed_data_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(Test_Processed_data_save_path), exist_ok=True)


    tf.data.Dataset.save(train_dataset,Train_Processed_data_save_path)
    tf.data.Dataset.save(test_dataset,Test_Processed_data_save_path)


def load_dataset():
    # Define the path for Loading the Processed_data
    Train_Processed_data_path = os.path.join("..", "data/Processed", "MNSIT_Train_Processed") 
    Test_Processed_data_path = os.path.join("..", "data/Processed", "MNSIT_Test_Processed") 

    # Ensure the directory exists
    os.makedirs(os.path.dirname(Train_Processed_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(Test_Processed_data_path), exist_ok=True)


    Train_Processed_data = tf.data.Dataset.load(Train_Processed_data_path)
    Test_Processed_data = tf.data.Dataset.load(Test_Processed_data_path)
    
    return Train_Processed_data, Test_Processed_data
    



