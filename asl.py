import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import pathlib

# Set page configuration
st.set_page_config(page_title="ASL Alphabet Detector", layout="centered")

# Set title
st.title("ASL Alphabet Detector")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("asl_model.keras")

model = load_model()

# Preprocess input image
def img_preprocess(image):
    img = image.convert("RGB")                                            # Convert image to RGB
    img = image.resize((200, 200), Image.Resampling.LANCZOS)              # Resize image to 200x200 pixels
    img_arr = tf.keras.preprocessing.image.img_to_array(img)              # Convert image to array
    img_arr = np.expand_dims(img_arr, axis=0)                             
    return img_arr


data_dir = pathlib.Path("asl_alphabet_train/asl_alphabet_train")          # Directory containing ASL alphabet images
class_names = sorted(os.listdir(data_dir)) 


# upload image
inp_img = st.file_uploader("Upload an image of ASL Alphabet", type=['jpg',"jpeg","png"])
    
# If image is uploaded
if inp_img:
    image = Image.open(inp_img)                                           # read image
    st.image(image, caption="Uploaded Image", use_container_width=True)   # Display the image

    image_arr = img_preprocess(image)                                     # preprocess the uploaded img
    
    st.write("Predicting...")

    model_pred = model.predict(image_arr)                                 # Predict the alphabet                         
    pred_index = np.argmax(model_pred)                                    # Get the index of the predicted class

    
    pred_alph = class_names[pred_index]                                   # Convert index to alphabet
    confidence = model_pred[0][pred_index]


    threshold = 0.6  # 60% confidence

    if confidence >= threshold:
        st.success(f"✅ Predicted Letter: **{pred_alph}**")
        st.write(f"🔍 Confidence: `{confidence * 100:.2f}%`")
    else:
        st.warning("⚠️ Not confident enough to predict a letter.")
        st.write(f"🔍 Confidence: `{confidence * 100:.2f}%`")
        


  


 

 


    