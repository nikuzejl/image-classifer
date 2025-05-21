import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os

# === Configuration ===
st.set_page_config(page_title="Image Classifier", layout="centered", page_icon="🖼️")
st.title("🖼️ Image Classifier")
st.markdown("Upload a CIFAR-10-like image to classify it into one of the 10 categories.")

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.success(f"Using device: `{device}`")

# === Class Labels ===
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# === Model Definition ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

# === Load Model ===
# @st.cache_resource
# def load_model(path="model.pth"):
#     model = SimpleCNN().to(device)
#     try:
#         state_dict = torch.load(path, map_location=device)
#         model.load_state_dict(state_dict)
#         model.eval()
#         return model
#     except Exception as e:
#         st.error(f"❌ Failed to load model: {e}")
#         st.stop()

# === Load Model ===
@st.cache_resource
def load_model(path="model.pth"):
    model = SimpleCNN().to(device)
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)  # Secure load
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()


if not os.path.exists("model.pth"):
    st.error("❌ Model file `model.pth` not found in current directory.")
    st.stop()

model = load_model("model.pth")

# === Image Upload ===
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1).cpu().numpy().flatten()
        top3_indices = probabilities.argsort()[-3:][::-1]

    # Display Top Prediction
    st.subheader("🪄 Prediction ✨")
    st.write(f"This is a **{classes[top3_indices[0]]}** with probability **{probabilities[top3_indices[0]]:.2%}**")

    # Display Top-3 Predictions
    st.markdown("## Top 3 Predictions")
    for idx in top3_indices:
        st.write(f"- {classes[idx]}: **{probabilities[idx]:.2%}**")
