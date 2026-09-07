#!/usr/bin/env python3
# src/export_onnx.py

import torch
import segmentation_models_pytorch as smp
import onnx
import onnxruntime as ort
import numpy as np
import os
import time
from pathlib import Path


def convert_to_onnx(checkpoint_path, onnx_path, input_size=(1, 3, 256, 256)):
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
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Model loaded from {checkpoint_path}")

    # Dummy input
    dummy_input = torch.randn(*input_size).to(device)

    # Export to ONNX
    print("🔄 Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,  # Changed from 14 to 11 for compatibility
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print(f"✅ ONNX model saved to {onnx_path}")

    # Verify the model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model verified")

    # Show size
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"📊 ONNX model size: {size_mb:.2f} MB")

    return onnx_path


def quantize_onnx(model_path, quantized_path):
    """
    Quantize ONNX model using onnxruntime quantization.
    """
    try:
        # Try the official quantization API
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        print("🔄 Quantizing model...")
        quantize_dynamic(
            model_input=model_path,
            model_output=quantized_path,
            per_channel=True,
            weight_type=QuantType.QInt8,
        )
        print(f"✅ Quantized model saved to {quantized_path}")

        # Show size reduction
        orig_size = os.path.getsize(model_path) / (1024 * 1024)
        quant_size = os.path.getsize(quantized_path) / (1024 * 1024)
        reduction = (1 - quant_size / orig_size) * 100
        print(f"📊 Size: {orig_size:.2f} MB → {quant_size:.2f} MB ({reduction:.1f}% reduction)")
        return quantized_path
        
    except ImportError:
        print("⚠️ Quantization library not available. Using non-quantized model.")
        return model_path
    except Exception as e:
        print(f"⚠️ Quantization failed: {e}")
        print("   Using non-quantized model instead.")
        return model_path


def test_inference(onnx_path, num_runs=50):
    """
    Test inference speed on CPU.
    """
    print(f"\n⚡ Testing inference speed...")
    
    # Use CPU for testing
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)

    # Warm-up
    for _ in range(5):
        session.run(None, {'input': dummy_input})

    # Measure
    times = []
    for _ in range(num_runs):
        start = time.time()
        session.run(None, {'input': dummy_input})
        times.append(time.time() - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    fps = 1000 / avg_time

    print(f"⚡ Inference time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"⚡ Frame rate: {fps:.2f} FPS")

    return avg_time, std_time, fps


def main():
    # Paths
    checkpoint_path = "checkpoints/best_model.pt"
    onnx_path = "model.onnx"
    quantized_path = "model_quantized.onnx"

    print("=" * 50)
    print("🔄 ONNX Export + Quantization")
    print("=" * 50)

    # Step 1: Export to ONNX
    print("\n📌 Step 1: Exporting to ONNX")
    try:
        convert_to_onnx(checkpoint_path, onnx_path)
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return

    # Step 2: Quantize (if possible)
    print("\n📌 Step 2: Quantizing")
    quantized_path = quantize_onnx(onnx_path, quantized_path)

    # Step 3: Test inference
    print("\n📌 Step 3: Testing Inference")
    test_inference(onnx_path)
    if os.path.exists(quantized_path):
        test_inference(quantized_path)

    print("\n" + "=" * 50)
    print("✅ ONNX export complete!")
    print(f"   Standard: {onnx_path}")
    print(f"   Quantized: {quantized_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
