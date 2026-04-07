import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time

# Page Config
st.set_page_config(
    page_title="AI Animal Classifier",
    page_icon="🐾",
    layout="wide"
)

# Theme Switch
theme = st.sidebar.radio(
    "🌙 Select Theme",
    ["Dark","Light"]
)

# Background Themes
if theme == "Dark":

    bg = """
    <style>
    .stApp {
    background-image: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color:white;
    }
    </style>
    """

else:

    bg = """
    <style>
    .stApp {
    background-image: linear-gradient(135deg,#f5f7fa,#c3cfe2);
    }
    </style>
    """

st.markdown(bg, unsafe_allow_html=True)

# Header
st.markdown(
"""
<h1 style='text-align:center;'>🐾 AI Animal Classifier Dashboard</h1>
<h4 style='text-align:center;'>Deep Learning Animal Detection System</h4>
""",
unsafe_allow_html=True
)

# Animal Classes
class_names = [
    "Dog 🐶",
    "Hen 🐔",
    "Horse 🐎",
    "Sheep 🐑"
]

# Animal Info Panel
animal_info = {
"Dog 🐶": "Dogs are loyal animals often kept as pets.",
"Hen 🐔": "Hens are domestic birds used for eggs.",
"Horse 🐎": "Horses are strong animals used for transport.",
"Sheep 🐑": "Sheep provide wool and meat."
}

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Load Model
@st.cache_resource
def load_model():

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)

    model.load_state_dict(
        torch.load("animal_model.pth", map_location="cpu")
    )

    model.eval()

    return model

model = load_model()

# Upload Section
uploaded_file = st.file_uploader(
    "📂 Upload Animal Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1,col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with st.spinner("🔍 Analyzing Image..."):

        time.sleep(2)

        img_tensor = transform(image).unsqueeze(0)

        outputs = model(img_tensor)

        probs = torch.nn.functional.softmax(
            outputs[0],
            dim=0
        )

        confidence, predicted = torch.max(probs,0)

    predicted_class = class_names[predicted]

    # Result Panel
    with col2:

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.progress(float(confidence))

        st.write(
            f"Confidence: {confidence*100:.2f}%"
        )

        # Animal Info
        st.info(
            animal_info[predicted_class]
        )

    # Probability Chart
    st.subheader("📊 Prediction Probabilities")

    fig, ax = plt.subplots()

    ax.bar(
        class_names,
        probs.detach().numpy()
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

    # Top Predictions
    st.subheader("🧠 Top Predictions")

    sorted_probs = sorted(
        zip(class_names,probs),
        key=lambda x: x[1],
        reverse=True
    )

    for name,prob in sorted_probs:

        st.write(
            f"{name} — {prob*100:.2f}%"
        )

# Footer
st.markdown(
"""
<hr>
<center>
Made by <b>Ibrarul Hassan</b> 🚀  
AI Animal Classification System
</center>
""",
unsafe_allow_html=True
)
