#!/usr/bin/env python3
"""
YOLO ONNX → K210 kmodel converter using nncase 1.x API.

Requires: nncase==1.9.0.20230322 (last v1.x with K210 support)
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def convert_onnx_to_kmodel(onnx_path, kmodel_path, target="k210",
                           input_size=224, quantize=True):
    """Convert ONNX model to K210 kmodel using nncase 1.x."""
    import nncase

    print(f"\n{'='*50}")
    print(f"  nncase {nncase.__version__}: ONNX → K210 kmodel")
    print(f"{'='*50}")
    print(f"  Target:      {target}")
    print(f"  Input size:  {input_size}")
    print(f"  Quantize:    {'INT8' if quantize else 'FP32'}")

    # Read ONNX
    with open(str(onnx_path), "rb") as f:
        onnx_data = f.read()
    print(f"  ONNX size:   {len(onnx_data) / 1024:.1f} KB")

    # Compile options (NHWC layout for K210 KPU)
    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.input_shape = [1, input_size, input_size, 3]
    compile_options.input_type = "float32"
    compile_options.input_layout = "NHWC"
    compile_options.output_layout = "NHWC"

    if quantize:
        compile_options.quant_type = "uint8"

    compiler = nncase.Compiler(compile_options)

    # Import ONNX
    print("\n[1/3] Importing ONNX...")
    import_options = nncase.ImportOptions()
    compiler.import_onnx(onnx_data, import_options)

    # PTQ calibration
    if quantize:
        print("[2/3] Setting up PTQ quantization...")
        ptq_options = nncase.PTQTensorOptions()
        ptq_options.samples_count = 5

        # Generate calibration data: [samples, H, W, C] float32
        calib_shape = [ptq_options.samples_count, input_size, input_size, 3]
        calib_data = np.random.rand(*calib_shape).astype(np.float32)
        ptq_options.set_tensor_data(calib_data.tobytes())

        compiler.use_ptq(ptq_options)

    # Compile
    print("[3/3] Compiling for K210 KPU...")
    compiler.compile()

    # Generate kmodel
    kmodel = compiler.gencode_tobytes()
    with open(str(kmodel_path), "wb") as f:
        f.write(kmodel)

    size_kb = kmodel_path.stat().st_size / 1024
    size_mb = size_kb / 1024

    print(f"\n{'='*50}")
    print(f"  kmodel generated successfully!")
    print(f"  Size: {size_kb:.1f} KB ({size_mb:.2f} MB)")
    print(f"  Path: {kmodel_path}")
    print(f"{'='*50}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="best.onnx")
    parser.add_argument("--output", "-o", default="best.kmodel")
    parser.add_argument("--target", default="k210")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()

    onnx_path = Path(args.input)
    kmodel_path = Path(args.output)

    if not onnx_path.exists():
        print(f"ERROR: {onnx_path} not found")
        sys.exit(1)

    try:
        convert_onnx_to_kmodel(
            onnx_path, kmodel_path,
            target=args.target,
            input_size=args.input_size,
            quantize=not args.no_quantize,
        )
    except Exception as e:
        print(f"\nERROR: Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Check size
    size_mb = kmodel_path.stat().st_size / (1024 * 1024)
    if size_mb > 6.0:
        print(f"WARNING: kmodel {size_mb:.1f}MB exceeds 6MB limit!")
        sys.exit(1)
    else:
        print(f"PASS: {size_mb:.1f}MB within 6MB limit")


if __name__ == "__main__":
    main()
