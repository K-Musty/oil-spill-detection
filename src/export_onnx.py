# src/export_onnx.py

import torch
import segmentation_models_pytorch as smp
import onnxruntime as ort
import numpy as np
import os
from pathlib import Path

def convert_to_onnx(checkpoint_path, onnx_path):
    """
    Convert PyTorch model to ONNX format.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    # Load model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=2,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, 256, 256).to(device)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print(f"✅ ONNX model saved to {onnx_path}")

    # Check the model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model verified")

    return onnx_path

def test_onnx_inference(onnx_path):
    """
    Test inference speed on CPU.
    """
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)

    import time
    times = []
    for _ in range(100):
        start = time.time()
        session.run(None, {'input': dummy_input})
        times.append(time.time() - start)

    avg_time = sum(times) / len(times) * 1000
    print(f"⚡ Average inference time: {avg_time:.2f} ms")
    print(f"   Frame rate: {1000/avg_time:.2f} FPS")

    return avg_time

if __name__ == "__main__":
    # Paths
    checkpoint_path = "checkpoints/best_model.pt"
    onnx_path = "model.onnx"

    # Convert
    convert_to_onnx(checkpoint_path, onnx_path)

    # Test speed
    test_onnx_inference(onnx_path)

    print("\n✅ ONNX export complete!")
    print(f"   Model saved to: {onnx_path}")
