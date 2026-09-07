# app/app.py

import os
import streamlit as st
import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import matplotlib.pyplot as plt

# ---------- PATH SETUP ----------
# Get the directory where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_model.pt")

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🛢️",
    layout="wide"
)

st.title("🛢️ Oil Spill Detection from SAR Imagery")
st.markdown("Upload a Sentinel-1 SAR image to detect oil spills using deep learning.")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    """Load the PyTorch model from the local path."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        return None, None

    device = torch.device('cpu')
    try:
        model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights=None,
            in_channels=3,
            classes=2,
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.to(device)
        model.eval()
        st.success("✅ Model loaded successfully!")
        return model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

model, device = load_model()
if model is None:
    st.stop()

# ---------- PREPROCESSING ----------
def preprocess_image(image):
    """Convert uploaded image to model input tensor."""
    image = np.array(image)
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    # Normalize to [0,1]
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    # Resize to 256x256
    img = Image.fromarray((image * 255).astype(np.uint8))
    img = img.resize((256, 256))
    image = np.array(img).astype(np.float32) / 255.0
    # Add batch and channel dimensions (1, 1, H, W)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return image.to(device)

# ---------- INFERENCE ----------
def run_inference(image_tensor):
    with torch.no_grad():
        pred = model(image_tensor)
        pred_mask = torch.argmax(pred, dim=1)[0].cpu().numpy()
    return pred_mask

# ---------- UI ----------
uploaded_file = st.file_uploader(
    "Choose a SAR image...",
    type=['png', 'jpg', 'jpeg', 'tif', 'tiff']
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", width=300)

    # Detection button
    if st.button("🔍 Detect Oil Spill", type="primary"):
        with st.spinner("Analyzing image..."):
            try:
                # Preprocess
                processed = preprocess_image(image)
                # Inference
                mask = run_inference(processed)
                oil_percentage = (mask == 1).sum() / mask.size * 100

                # Prepare display image
                original_display = processed[0, 0].cpu().numpy()

                # Create side‑by‑side plot
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(original_display, cmap='gray')
                axes[0].set_title('Original SAR Image')
                axes[0].axis('off')

                axes[1].imshow(original_display, cmap='gray')
                axes[1].imshow(mask, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
                axes[1].set_title(f'Oil Detection ({oil_percentage:.1f}%)')
                axes[1].axis('off')

                plt.tight_layout()

                with col2:
                    st.pyplot(fig)

                    # Status
                    if oil_percentage > 10:
                        st.error(f"⚠️ HIGH oil concentration: {oil_percentage:.1f}%")
                    elif oil_percentage > 2:
                        st.warning(f"⚠️ Oil detected: {oil_percentage:.1f}%")
                    else:
                        st.success(f"✅ No significant oil: {oil_percentage:.1f}%")

            except Exception as e:
                st.error(f"Error during inference: {e}")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Built with PyTorch, segmentation‑models‑pytorch, and Streamlit.")
