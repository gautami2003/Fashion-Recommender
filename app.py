import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st
from PIL import Image

# Streamlit UI
st.title("Fashion Recommendation System")
st.write("Upload an image to get similar fashion recommendations.")

# Load Features & Filenames Safely
features_path = "Images_features.pkl"
filenames_path = "filenames.pkl"

try:
    with open(features_path, 'rb') as f:
        Image_features = pkl.load(f)
    with open(filenames_path, 'rb') as f:
        filenames = pkl.load(f)
except Exception as e:
    st.error(f"Error loading data files: {e}")
    st.stop()

# Ensure filenames are valid
filenames = [f for f in filenames if os.path.exists(f)]

# Cache Model to Prevent Redownloading
@st.cache_resource
def load_model():
    try:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        return tf.keras.models.Sequential([base_model, GlobalMaxPool2D()])
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# Train Nearest Neighbors Model
neighbors = NearestNeighbors(n_neighbors=min(6, len(Image_features)), algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# Feature Extraction Function
def extract_features_from_images(image_path, model):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_expand_dim = np.expand_dims(img_array, axis=0)
        img_preprocess = preprocess_input(img_expand_dim)
        result = model.predict(img_preprocess).flatten()
        return result / norm(result)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# File Upload Handler
upload_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if upload_file is not None:
    # Ensure Upload Directory Exists
    if not os.path.exists("upload"):
        os.makedirs("upload")

    # Save Uploaded File
    image_path = os.path.join("upload", upload_file.name)
    with open(image_path, 'wb') as f:
        f.write(upload_file.getbuffer())

    st.subheader("Uploaded Image")
    st.image(upload_file, width=250)

    # Extract Features & Find Recommendations
    input_img_features = extract_features_from_images(image_path, model)
    
    if input_img_features is not None:
        distances, indices = neighbors.kneighbors([input_img_features])
        
        # Ensure we do not exceed available images
        num_recommendations = min(5, len(indices[0]) - 1)
        
        if num_recommendations > 0:
            st.subheader("Recommended Images")
            cols = st.columns(num_recommendations)
            
            for i in range(num_recommendations):
                recommended_index = indices[0][i + 1]  # Skip the first index as it is the input image
                # Ensure recommended_index is within bounds
                if recommended_index < len(filenames):
                    file_path = filenames[recommended_index]
                    # Check if the recommended image exists
                    if os.path.exists(file_path):
                        cols[i].image(file_path, width=150)
                    else:
                        st.warning(f"Missing image: {file_path}")
                else:
                    st.warning(f"Index {recommended_index} is out of bounds for filenames list.")
        else:
            st.warning("No similar images found. Try uploading another image.")

    # Cleanup: Remove Temporary File
    os.remove(image_path)
