# Re-export the ONNX model correctly

import torch
import segmentation_models_pytorch as smp
import os

# Load model
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights=None,
    in_channels=3,
    classes=2,
)
model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location='cpu'))
model.eval()
print("✅ Model loaded")

# Create dummy input
dummy = torch.randn(1, 3, 256, 256)

# Export with opset=11 (most stable)
torch.onnx.export(
    model,
    dummy,
    'model.onnx',
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
print("✅ ONNX export complete")

# Check file size
size_mb = os.path.getsize('model.onnx') / (1024 * 1024)
print(f"📊 Model size: {size_mb:.2f} MB")


