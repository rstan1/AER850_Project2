"""
AER 850 Project 2
Created on Thu Nov  7 11:46:25 2024
@author: robstan 501095883
"""

#Importing libraries
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Step 5: Model Testing

# Define paths to the test images and true labels
test_image_paths = {
    "crack": "Data/test/crack/test_crack.jpg",
    "missing-head": "Data/test/missing-head/test_missinghead.jpg",
    "paint-off": "Data/test/paint-off/test_paintoff.jpg"
}

# True labels for each image
true_labels = {
    "crack": "crack",
    "missing-head": "missing-head",
    "paint-off": "paint-off"
}

# Load the trained model
model = load_model('trained_model.h5')  

# Class labels
class_labels = ["crack", "missing-head", "paint-off"]

# Function to preprocess and predict a single image
def predict_image(image_path, true_label, model):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(100, 100))  
    img_array = img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)[0]
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[predicted_class]
    
    # Display the image with true and predicted labels
    plt.figure(figsize=(6, 6))
    plt.imshow(load_img(image_path))
    plt.axis('off')
    plt.title(f"True Crack Classification Label: {true_label}\n"
              f"Predicted Crack Classification Label: {predicted_label}")
    
    # Display probabilities below the image
    plt.figtext(
        0.5, -0.1,  
        "\n".join([f"{label.capitalize()}: {predictions[i] * 100:.1f}%" for i, label in enumerate(class_labels)]),
        ha="center", color="green", fontsize=12, fontweight="bold", wrap=True
    )

    plt.show()

# Run predictions on each test image
for label, path in test_image_paths.items():
    predict_image(path, true_labels[label], model)

