# app.py - Gradio version for Hugging Face Spaces

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt

# Load ONNX model
session = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])

def preprocess_image(image):
    """Preprocess uploaded image for inference."""
    # Convert to numpy
    image = np.array(image)
    
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Resize to 256x256
    img = Image.fromarray((image * 255).astype(np.uint8))
    img = img.resize((256, 256))
    image = np.array(img).astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    image = image[np.newaxis, np.newaxis, :, :]
    return image

def run_inference(image):
    """Run ONNX model on preprocessed image."""
    outputs = session.run(None, {'input': image})
    pred = outputs[0]
    pred_mask = np.argmax(pred, axis=1)[0]
    return pred_mask

def detect_oil(image):
    """Main function called by Gradio."""
    # Preprocess
    processed = preprocess_image(image)
    
    # Run inference
    mask = run_inference(processed)
    
    # Calculate oil percentage
    oil_percentage = (mask == 1).sum() / mask.size * 100
    
    # Create overlay
    original_display = processed[0, 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original
    axes[0].imshow(original_display, cmap='gray')
    axes[0].set_title('Original SAR Image')
    axes[0].axis('off')
    
    # Overlay
    axes[1].imshow(original_display, cmap='gray')
    axes[1].imshow(mask, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title(f'Oil Detection ({oil_percentage:.1f}%)')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Status text
    if oil_percentage > 10:
        status = f"⚠️ HIGH oil concentration: {oil_percentage:.1f}%"
    elif oil_percentage > 2:
        status = f"⚠️ Oil detected: {oil_percentage:.1f}%"
    else:
        status = f"✅ No significant oil: {oil_percentage:.1f}%"
    
    return fig, status

# Gradio interface
title = "🛢️ Oil Spill Detection from SAR Imagery"
description = """
Upload a Sentinel-1 SAR image to detect oil spills using deep learning.

**Model**: U-Net with ResNet34 backbone trained on the Eastern Mediterranean Oil Slick Dataset.
**Author**: Abdulrahman Kalli Mustapha
"""

interface = gr.Interface(
    fn=detect_oil,
    inputs=gr.Image(type="pil", label="Upload SAR Image"),
    outputs=[
        gr.Plot(label="Detection Result"),
        gr.Textbox(label="Status")
    ],
    title=title,
    description=description,
    theme="soft"
)

if __name__ == "__main__":
    interface.launch()
