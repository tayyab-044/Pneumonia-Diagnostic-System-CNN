import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import time
import os

# Set page config
st.set_page_config(
    page_title="Pneumonia Detection from Chest X-rays",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model (replace with your actual model loading code)
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('pneumonia_cnn_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Function to preprocess image
def preprocess_image(img):
    # Convert to RGB if grayscale
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((150, 150))  # Match model's expected input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to make prediction
def predict_pneumonia(img):
    """
    Predicts pneumonia from a preprocessed X-ray image.
    
    Returns:
        result (str)       : "Pneumonia Detected" or "Normal (No Pneumonia)"
        pneu_conf (float)  : Confidence (%) for Pneumonia
        norm_conf (float)  : Confidence (%) for Normal
    """
    # Preprocess image as required by your model
    processed_img = preprocess_image(img)  # Make sure this returns shape (1, H, W, C)
    
    # Model prediction: probability of pneumonia (0–1)
    prob_pneumonia = float(model.predict(processed_img)[0][0])
    prob_normal = 1 - prob_pneumonia
    
    # Convert to percentages
    pneu_conf = prob_pneumonia * 100
    norm_conf = prob_normal * 100
    
    # Decide result label
    result = "Pneumonia Detected" if prob_pneumonia > 0.5 else "Normal (No Pneumonia)"
    
    return result, pneu_conf, norm_conf


# App header
st.title("🩺 Pneumonia Detection from Chest X-rays")
st.markdown("""
This application uses a deep learning model to detect pneumonia from chest X-ray images. 
Upload a chest X-ray image, and the system will analyze it for signs of pneumonia.
""")

st.markdown("---")

# ---- Section: Access Sample Images ----
st.markdown("### 📂 Access Sample Images for Testing")

st.info(
    "You can explore sample chest X-ray images stored in our public Google Drive folder. "
    "Click the button below to open the folder in a new tab and download any image to test the model."
)

# 🔗 Replace this link with your actual Google Drive folder link
drive_link = "https://drive.google.com/drive/folders/1knXuROKjyALelpe_zI6rg2LeNH90PZv6?usp=drive_link"

# ---- Create a button that opens the Drive folder in a new tab ----
st.markdown(
    f"""
    <a href="{drive_link}" target="_blank">
        <button style="
            background-color:#4CAF50;
            color:white;
            padding:10px 20px;
            border:none;
            border-radius:5px;
            font-size:16px;
            cursor:pointer;
        ">🔗 Open Sample Images Folder</button>
    </a>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Chest X-ray")
    uploaded_file = st.file_uploader(
        "Choose an X-ray image...", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="Upload a clear chest X-ray image for analysis"
    )
    
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded X-ray", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")

with col2:
    st.subheader("Analysis Results")
    
    if uploaded_file is not None:
        with st.spinner("Analyzing X-ray..."):
            # Simulate processing time for better UX
            time.sleep(1)
            
            # Get prediction and probabilities
            result, pneu_conf, norm_conf = predict_pneumonia(img)
            
            if result is not None:
                # Animated result display
                with st.empty():
                    for percent_complete in range(100):
                        time.sleep(0.01)
                        st.progress(percent_complete + 1)
                
                # Display text results
                if "Pneumonia" in result:
                    st.error(f"## {result}")
                    st.error(f"### Confidence: {pneu_conf:.2f}%")
                else:
                    st.success(f"## {result}")
                    st.success(f"### Confidence: {norm_conf:.2f}%")
                
                # Bar chart visualization
                fig, ax = plt.subplots()
                labels = ['Pneumonia', 'Normal']
                values = [pneu_conf, norm_conf]       # red = Pneumonia, green = Normal
                colors = ['#FF5252', '#4CAF50']
                
                ax.bar(labels, values, color=colors)
                ax.set_ylim(0, 100)
                ax.set_ylabel('Confidence (%)')
                ax.set_title('Detection Confidence')
                
                st.pyplot(fig)
            else:
                st.warning("Could not process the image. Please try another image.")


st.markdown("<br><br>", unsafe_allow_html=True)

# -----------------------------
# Disclaimer Section
# -----------------------------st.markdown("---")

st.markdown("""
**Disclaimer:** This Diagnostic System uses AI-based image analysis for research and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. 
Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
""")

# -----------------------------
# Copyright / Footer Section
# -----------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 14px;'>
        © 2025 Pneumonia Diagnostic System. All rights reserved.  
        <br>Developed by <b>Tayyab</b>.
    </div>
    """,
    unsafe_allow_html=True
)