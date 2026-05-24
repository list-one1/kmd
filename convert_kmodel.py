#!/usr/bin/env python3
"""
YOLO ONNX → K210 kmodel conversion script
Uses nncase to compile ONNX to K210-compatible kmodel format.

Usage: python convert_kmodel.py [--input best.onnx] [--output model.kmodel]
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def check_onnx(onnx_path):
    """Print ONNX model info."""
    import onnx
    model = onnx.load(str(onnx_path))
    print("=== ONNX Model Info ===")
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  Input:  {inp.name}  shape={shape}")
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name}  shape={shape}")
    print(f"  Opset:  {model.opset_import[0].version}")


def convert_to_kmodel(onnx_path, kmodel_path, target="k210",
                      input_shape=(1, 3, 320, 320),
                      quantize=True, calibrate_dataset=None):
    """
    Convert ONNX to K210 kmodel using nncase.

    Args:
        onnx_path: Path to ONNX model
        kmodel_path: Output kmodel path
        target: Target chip ('k210')
        input_shape: Input shape for the model (NCHW)
        quantize: Enable INT8 quantization
        calibrate_dataset: Path to calibration images (npy format)
    """
    try:
        import nncase
    except ImportError:
        print("ERROR: nncase not installed. Run: pip install nncase")
        return False

    print(f"\n=== Converting ONNX → kmodel (target={target}) ===")
    print(f"  Input shape: {input_shape}")
    print(f"  Quantize:    {quantize}")

    # Step 1: Create compiler
    compiler_options = nncase.CompileOptions()
    compiler_options.target = target
    compiler_options.input_type = "uint8" if quantize else "float32"
    compiler_options.input_shape = list(input_shape)
    compiler_options.input_range = [0.0, 1.0]
    compiler_options.preprocess = True
    compiler_options.input_layout = "NCHW"
    compiler_options.output_layout = "NCHW"

    if quantize:
        compiler_options.quant_type = "uint8"
        compiler_options.w_quant_type = "uint8"

    compiler = nncase.Compiler(compiler_options)

    # Step 2: Import ONNX
    print("\n[1/3] Importing ONNX model...")
    with open(str(onnx_path), "rb") as f:
        onnx_data = f.read()

    compiler.import_onnx(onnx_data)

    # Step 3: Set quantize output
    if quantize:
        compiler.use_ptq()

    # Step 4: Compile
    print("[2/3] Compiling for K210 NPU...")
    compiler.compile()

    # Step 5: Generate kmodel
    print("[3/3] Generating kmodel file...")
    with open(str(kmodel_path), "wb") as f:
        f.write(compiler.gencode())

    size_kb = Path(kmodel_path).stat().st_size / 1024
    print(f"\n✓ kmodel generated: {kmodel_path}")
    print(f"  Size: {size_kb:.1f} KB")

    return True


def main():
    parser = argparse.ArgumentParser(description="ONNX → K210 kmodel converter")
    parser.add_argument("--input", "-i", default="best.onnx",
                        help="Input ONNX model path")
    parser.add_argument("--output", "-o", default="best.kmodel",
                        help="Output kmodel path")
    parser.add_argument("--target", default="k210",
                        help="Target chip (k210)")
    parser.add_argument("--input-size", type=int, default=320,
                        help="Input size (square)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Disable INT8 quantization")
    args = parser.parse_args()

    onnx_path = Path(args.input)
    kmodel_path = Path(args.output)

    if not onnx_path.exists():
        print(f"ERROR: ONNX model not found: {onnx_path}")
        sys.exit(1)

    check_onnx(onnx_path)

    input_shape = (1, 3, args.input_size, args.input_size)
    success = convert_to_kmodel(
        onnx_path, kmodel_path,
        target=args.target,
        input_shape=input_shape,
        quantize=not args.no_quantize,
    )

    if not success:
        print("\nConversion failed!")
        sys.exit(1)

    # Verify size limits
    size_mb = kmodel_path.stat().st_size / (1024 * 1024)
    if size_mb > 6.0:
        print(f"\n⚠ WARNING: kmodel size ({size_mb:.1f} MB) exceeds 6MB limit!")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
