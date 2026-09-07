#!/usr/bin/env python3
# src/export_onnx.py

import torch
import segmentation_models_pytorch as smp
import onnx
import onnxruntime as ort
import numpy as np
import os
import time

def convert_to_onnx(checkpoint_path, onnx_path):
    """Convert PyTorch model to ONNX."""
    
    # Load model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=2,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    print(f"✅ Model loaded from {checkpoint_path}")

    # Dummy input
    dummy_input = torch.randn(1, 3, 256, 256)

    # Export to ONNX
    print("🔄 Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✅ ONNX model saved to {onnx_path}")

    # Verify
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model verified")

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"📊 Model size: {size_mb:.2f} MB")

    return onnx_path


def test_inference(onnx_path):
    """Test inference speed."""
    print("\n⚡ Testing inference...")
    
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)

    # Warm-up
    for _ in range(5):
        session.run(None, {'input': dummy_input})

    # Measure
    times = []
    for _ in range(50):
        start = time.time()
        session.run(None, {'input': dummy_input})
        times.append(time.time() - start)

    avg_time = np.mean(times) * 1000
    print(f"⚡ Average inference time: {avg_time:.2f} ms")
    print(f"⚡ Frame rate: {1000/avg_time:.2f} FPS")


if __name__ == "__main__":
    checkpoint_path = "checkpoints/best_model.pt"
    onnx_path = "model.onnx"

    convert_to_onnx(checkpoint_path, onnx_path)
    test_inference(onnx_path)

    print("\n✅ ONNX export complete!")
    print(f"   Model: {onnx_path}")
