#!/usr/bin/env python3
"""
YOLO ONNX → K210 kmodel converter using nncase 2.x API.
"""

import argparse
import sys
from pathlib import Path


def convert_onnx_to_kmodel(onnx_path, kmodel_path, target="k210",
                           input_shape=(1, 3, 224, 224), quantize=True):
    """Convert ONNX model to K210 kmodel using nncase."""
    import nncase

    print(f"\n{'='*50}")
    print(f"  nncase {nncase.__version__}: ONNX → K210 kmodel")
    print(f"{'='*50}")
    print(f"  Target:      {target}")
    print(f"  Input shape: {input_shape}")
    print(f"  Quantize:    {'INT8' if quantize else 'FP32'}")

    # Read ONNX
    with open(str(onnx_path), "rb") as f:
        onnx_data = f.read()
    print(f"  ONNX size:   {len(onnx_data) / 1024:.1f} KB")

    # Create compiler with compile options
    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.input_shape = list(input_shape)

    if quantize:
        compile_options.input_type = "uint8"
        compile_options.input_range = [0, 255]
    else:
        compile_options.input_type = "float32"
        compile_options.input_range = [0, 1]

    compile_options.preprocess = True
    compile_options.input_layout = "NCHW"
    compile_options.output_layout = "NCHW"
    compile_options.dump_ir = False

    compiler = nncase.Compiler(compile_options)

    # Import model
    print("\n[1/3] Importing ONNX...")
    compiler.import_onnx(onnx_data)

    # Quantize
    if quantize:
        print("[2/3] Setting up PTQ quantization...")
        compiler.use_ptq()

    # Compile
    print("[3/3] Compiling for K210 KPU...")
    compiler.compile()

    # Generate kmodel
    code = compiler.gencode(str(kmodel_path))
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
            input_shape=(1, 3, args.input_size, args.input_size),
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
