import tensorflow as tf
import numpy as np
from PIL import Image
import os

def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def preprocess_fn(images, labels, batch_size=32, shuffle=True, augment=False, flatten=False):

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    # Reshape the input data to match the expected shape
    dataset = dataset.map(lambda x, y: (tf.reshape(x, (-1, 28, 28, 1)), y))

    # Shuffle the dataset if required
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
        
    # Apply augmentation if required
    if augment:
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)        
        
    # flatten the dataset if required
    if flatten:
        dataset = dataset.map(lambda image, label: (tf.reshape(image, [-1]), label), num_parallel_calls=tf.data.AUTOTUNE)
        
    # Apply normalization (and flattening if required)
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
# Augmentation function 
def augment_fn(image, label):
    # Ensure the image is in float32 format and add a channel dimension
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=-1)
    
    # Random rotation (reduced probability)
    if tf.random.uniform(()) < 0.25:
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    
    # Random flip left-right (reduced probability)
    if tf.random.uniform(()) < 0.25:
        image = tf.image.random_flip_left_right(image)
    
    # Random brightness adjustment (reduced range)
    image = tf.image.random_brightness(image, max_delta=0.05)
    
    # Random contrast adjustment (narrower range)
    image = tf.image.random_contrast(image, lower=0.95, upper=1.05)
    
    # Ensure pixel values are still in [0, 1] range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    # Remove the added channel dimension
    image = tf.squeeze(image, axis=-1)
    return image, label



def preprocess_predicted_image(image_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The file {image_path} does not exist.")
    
    try:
        # Load the image
        img = Image.open(image_path)
    except OSError as e:
        raise OSError(f"Error opening the image file: {e}")
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize to 28x28 pixels
    img = img.resize((28, 28))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Invert the colors (255 - pixel_value)
    img_array = 255 - img_array

    # Convert to tensor
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    # Normalize pixel values
    img_tensor = img_tensor / 255.0
    
    # Flatten the image
    img_tensor = tf.reshape(img_tensor, (1, 784))
    
    return img_tensor