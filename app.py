import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the model (adjust the path to your model)
classifier = load_model('brain_tumor_model.h5')  # Replace with your model's path

# Streamlit Web App Interface
st.title("Brain Tumor MRI Classification")
st.write("Upload an MRI image to classify the tumor type.")

# File uploader
uploaded_image = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open the uploaded image
    test_image = Image.open(uploaded_image)
    
    # Data Preprocessing
    test_image = test_image.resize((64, 64))  # Resize to match the model's input size
    test_image = np.array(test_image)  # Convert image to numpy array
    test_image = test_image / 255.0  # Normalize the image (if this was done during training)
    test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

    # Prediction
    result = classifier.predict(test_image)

    # Debugging: Check the raw output from the model
    st.write("Model Prediction (Raw Output):", result)

    # Get the predicted class
    predicted_class = np.argmax(result, axis=1)
    
    # Debugging: Check the predicted class
    st.write("Predicted Class Index:", predicted_class)

    # Display the result based on predicted class
    if predicted_class == 0:
        st.image(uploaded_image, caption="Glioma Tumor", use_column_width=True)
        st.write("The image is classified as a Glioma Tumor.")
    elif predicted_class == 1:
        st.image(uploaded_image, caption="Meningioma Tumor", use_column_width=True)
        st.write("The image is classified as a Meningioma Tumor.")
    elif predicted_class == 2:
        st.image(uploaded_image, caption="No Tumor", use_column_width=True)
        st.write("The image is classified as No Tumor.")
    else:
        st.image(uploaded_image, caption="Pituitary Tumor", use_column_width=True)
        st.write("The image is classified as a Pituitary Tumor.")
