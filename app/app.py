# app.py - PyTorch version (NO ONNX)

import gradio as gr
import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import matplotlib.pyplot as plt

# Load PyTorch model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights=None,
    in_channels=3,
    classes=2,
)
model.load_state_dict(torch.load('best_model.pt', map_location='cpu'))
model.to(device)
model.eval()
print("✅ Model loaded")

def preprocess_image(image):
    """Preprocess uploaded image for inference."""
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
    """Run PyTorch model on preprocessed image."""
    with torch.no_grad():
        pred = model(image)
        pred_mask = torch.argmax(pred, dim=1)[0].cpu().numpy()
    return pred_mask

def detect_oil(image):
    """Main function called by Gradio."""
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
    
    status = "⚠️ Oil detected" if oil_percentage > 2 else "✅ No significant oil"
    return fig, status

interface = gr.Interface(
    fn=detect_oil,
    inputs=gr.Image(type="pil", label="Upload SAR Image"),
    outputs=[gr.Plot(label="Detection Result"), gr.Textbox(label="Status")],
    title="🛢️ Oil Spill Detection from SAR Imagery",
    description="Upload a Sentinel-1 SAR image to detect oil spills using deep learning."
)

if __name__ == "__main__":
    interface.launch()
