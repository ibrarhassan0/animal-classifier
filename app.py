import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import requests

st.title("🐾 RGB Animal Classifier")

device = torch.device("cpu")

# ==============================
# Model Download
# ==============================

MODEL_URL = "https://huggingface.co/ihassa074/animal-classifier-model/resolve/main/animal_model.pth?download=true"
MODEL_PATH = "animal_model.pth"

if not os.path.exists(MODEL_PATH):
    st.write("Downloading model... Please wait ⏳")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    st.write("Model downloaded successfully ✅")

# ==============================
# Load Model
# ==============================

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 4 classes
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# ==============================
# Image Transform
# ==============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ⚠️ MUST MATCH TRAINING ORDER (alphabetical)
classes = ["dog", "hen", "horse", "sheep"]

# ==============================
# Upload Image
# ==============================

file = st.file_uploader("Upload an animal image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    st.success(f"Prediction: {classes[predicted.item()]}")
    st.write(f"Confidence: {confidence.item():.2f}")
