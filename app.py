import streamlit as st
import tensorflow as tf
import numpy as np 
from PIL import Image


#page configuration

st.set_page_config(
    page_title = "Nature  Image Classification",
    page_icon = "🌍",
    layout = "centered"
    )


# Custom UI styling
st.markdown("""
<style>

/* Main application background */
.stApp {
    background: linear-gradient(
        135deg,
        #e3f2fd,
        #f5f9ff,
        #e8eaf6
    );
}

/* Main content area */
.block-container {
    padding-top: 2rem;
    max-width: 900px;
}

/* All general text */
.stApp,
.stApp p,
.stApp label,
.stApp span,
.stApp li {
    color: #172b4d;
    font-weight: 500;
}

/* Main title */
h1 {
    color: #102a43 !important;
    text-align: center;
    font-weight: 800 !important;
    letter-spacing: 0.5px;
}

/* Subheadings */
h2, h3 {
    color: #163a5f !important;
    font-weight: 700 !important;
}

/* Description text */
.stMarkdown p {
    color: #243b53 !important;
    font-size: 16px;
    font-weight: 500;
}

/* Buttons */
.stButton > button {
    background-color: #1976d2;
    color: white !important;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1.5rem;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #0d47a1;
    color: white !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.85);
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #9bbce0;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] section {
    color: #17365d !important;
    font-weight: 600 !important;
}

/* Prediction result */
.stAlert p {
    color: #14532d !important;
    font-weight: 700 !important;
}

/* General input text */
.stTextInput input {
    color: #102a43 !important;
}


/* File uploader - white background */
[data-testid="stFileUploader"] {
    background-color: #ffffff !important;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #c5d9f0;
}

/* Upload area text */
[data-testid="stFileUploader"] section {
    background-color: #ffffff !important;
    color: #17365d !important;
}

/* Browse files button */
[data-testid="stFileUploader"] button {
    background-color: #1976d2 !important;
    color: #ffffff !important;
    border: none;
}

/* Predicted class - white text */
.stAlert,
.stAlert p,
.stAlert div,
.stAlert span {
    color: #ffffff !important;
    font-weight: 700 !important;
}

</style>
""", unsafe_allow_html=True)


#Classes names (Same order we used for training)
class_names = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]    

#Load the trained model
st.info("Starting application...")

@st.cache_resource
def load_model():
    print("Loading EfficientNetB0 model...", flush=True)

    model = tf.keras.models.load_model(
        "best_efficientnetb0.keras",
        compile=False
    )

    print("EfficientNetB0 model loaded successfully!", flush=True)

    return model

st.info("Loading EfficientNetB0 model...")

model = load_model()

st.success("Model loaded successfully!")

#Application title
st.title("🌍 Nature Image Classification")
st.write(
       "Upload an image to identify whether it belongs "
        "to buildings, forest, glacier, mountain, sea, or street."
)

#upload image

uploaded_file = st.file_uploader(
    "choose an image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption = "Uploaded File", use_container_width = True)

    #processes  image
    resized_image = image.resize((180,180))
    image_array = np.array(resized_image)
    image_array = np.expand_dims(image_array, axis = 0)

    #Predict

    if st.button("Classify Image"):
        predictions = model.predict(image_array, verbose = 0)

        predicted_index =  np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence =  predictions[0][predicted_index] *100

        st.markdown(
            f"<p style='color: black; font-size: 20px; font-weight: bold;'>"
            f"Predicted class: {predicted_class.title()}</p>",
            unsafe_allow_html=True
        )
        st.write(f"Confidence : {confidence:.2f}%")

        #Display probabilities
        st.subheader("Prediction  Probabilities")

        for i, class_name in  enumerate(class_names):
            st.write(
                f"{class_name.title()}: "
                f"{predictions[0][i]*100:.2f}%"
            )

