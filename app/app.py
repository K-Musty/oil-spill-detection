# streamlit_app.py

import streamlit as st
import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Oil Spill Detection", layout="wide")
st.title("🛢️ Oil Spill Detection from SAR Imagery")
st.markdown("Upload a Sentinel-1 SAR image to detect oil spills.")

@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=3,
        classes=2,
    )
    model.load_state_dict(torch.load('best_model.pt', map_location='cpu'))
    model.to(device)
    model.eval()
    return model, device

try:
    model, device = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

def preprocess_image(image):
    image = np.array(image)
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    img = Image.fromarray((image * 255).astype(np.uint8))
    img = img.resize((256, 256))
    image = np.array(img).astype(np.float32) / 255.0
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return image.to(device)

def run_inference(image):
    with torch.no_grad():
        pred = model(image)
        pred_mask = torch.argmax(pred, dim=1)[0].cpu().numpy()
    return pred_mask

uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", width=300)
    
    if st.button("🔍 Detect Oil Spill", type="primary"):
        with st.spinner("Analyzing image..."):
            processed = preprocess_image(image)
            mask = run_inference(processed)
            oil_percentage = (mask == 1).sum() / mask.size * 100
            
            original_display = processed[0, 0].cpu().numpy()
            
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
                
                if oil_percentage > 10:
                    st.error(f"⚠️ HIGH oil concentration: {oil_percentage:.1f}%")
                elif oil_percentage > 2:
                    st.warning(f"⚠️ Oil detected: {oil_percentage:.1f}%")
                else:
                    st.success(f"✅ No significant oil: {oil_percentage:.1f}%")
