import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os
import random
from PIL import Image
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Project details
def show_project_details():
    st.title("DEPT OF INFORMATION TECHNOLOGY")
    st.subheader("NATIONAL INSTITUTE OF TECHNOLOGY")
    st.markdown("### COURSE PROJECT TITLE")
    st.markdown("### ACADEMIC SESSION \"JAN-APRIL 2025\"")
    st.markdown("#### CARRIED OUT BY- Chaitanya Gulhane 221AI015 AND Gagan Deepankar 221AI019")


# Load the trained model
model = load_model("ensemble_model.h5", compile=False)

# Define class labels
class_labels = [
    'early_leaf_spot_1',
    'early_rust_1',
    'healthy_leaf_1',
    'late_leaf_spot_1',
    'nutrition_deficiency_1',
    'rust_1'
]

# Function to predict an image
def predict_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        st.error("Error: Unable to read image.")
        return None, None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    predictions = model.predict(image)
    probabilities = tf.nn.softmax(predictions).numpy()
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[0][predicted_class]
    
    return class_labels[predicted_class], confidence

dataset_dir = r"C:\Users\Chaitanya\Desktop\DL_project\DL221AI015\DL221AI015\DLai015\Dataset of groundnut plant leaf images for classification and detection\Dataset of groundnut plant leaf images for classification and detection\Raw_Data\predict"

show_project_details()

# # Check if path exists
# if not os.path.exists(dataset_dir):
#     #print("Error: The dataset directory does not exist!")
#     pass

# # List all files in the directory
# for root, _, files in os.walk(dataset_dir):
#     #print(f"Checking directory: {root}")
#     for file in files:
#         #print(file)
#         pass
#     pass

all_images = []
for root, _, files in os.walk(dataset_dir):
    #st.write(f"Checking directory: {root}")
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            all_images.append(os.path.join(root, file))

#st.write(f"Total images found: {len(all_images)}")

# if not all_images:
#     st.error("No images found in the dataset directory!")

# if all_images:
#     random_image_path = random.choice(all_images)
#     predicted_class, confidence = predict_image(random_image_path)
    
#     # Display image and result
#     st.image(Image.open(random_image_path), caption="Selected Image", use_column_width=True)
#     st.write(f"**Prediction:** {predicted_class}  \n**Confidence:** {confidence:.2f}")
# else:
#     st.error("No images found in the dataset directory!")

def run():
    random_image_path = random.choice(all_images)
    predicted_class, confidence = predict_image(random_image_path)
    
    # Display image and result
    st.image(Image.open(random_image_path), caption="Selected Image", use_column_width=True)
    st.write(f"**Prediction:** {predicted_class}  \n**Confidence:** {confidence:.2f}")

st.button('Select a Random image and predict', on_click=run)