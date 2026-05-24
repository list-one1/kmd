#!/usr/bin/env python3
"""
YOLO ONNX → K210 kmodel converter using nncase 1.x API.

Based on the working pytorch-k210 conversion pipeline.
Requires: nncase==1.9.0.20230322, onnxsim, onnx
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def simplify_onnx(onnx_path):
    """Run ONNX shape inference + simplifier to make model nncase-compatible."""
    import onnx
    import onnxsim

    print("  Running ONNX simplification...")
    onnx_model = onnx.load(str(onnx_path))

    # Infer shapes first
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    # Auto-detect input shapes
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    input_names = list(set(input_all) - set(input_initializer))
    input_tensors = [node for node in onnx_model.graph.input if node.name in input_names]

    input_shapes = {}
    for e in input_tensors:
        onnx_type = e.type.tensor_type
        shape = []
        for i, d in enumerate(onnx_type.shape.dim):
            val = d.dim_value if d.dim_value != 0 else 1
            shape.append(val)
        input_shapes[e.name] = shape
        print(f"  Input: {e.name} shape={shape}")

    onnx_model, check = onnxsim.simplify(onnx_model, input_shapes=input_shapes)
    if not check:
        print("  WARNING: ONNX simplification failed validation, continuing anyway")

    simplified_path = onnx_path.parent / (onnx_path.stem + "_simplified.onnx")
    onnx.save_model(onnx_model, str(simplified_path))
    print(f"  Simplified: {simplified_path} ({simplified_path.stat().st_size / 1024:.1f} KB)")
    return simplified_path


def convert_onnx_to_kmodel(onnx_path, kmodel_path, target="k210", quantize=True):
    """Convert ONNX model to K210 kmodel."""
    import nncase

    print(f"\n{'='*50}")
    print(f"  nncase {nncase.__version__}: ONNX -> {target} kmodel")
    print(f"{'='*50}")
    print(f"  Input:  {onnx_path}")
    print(f"  Output: {kmodel_path}")
    print(f"  Quantize: {'INT8' if quantize else 'FP32'}")

    # Step 0: Simplify ONNX
    simplified_path = simplify_onnx(onnx_path)

    # Read simplified ONNX
    with open(str(simplified_path), "rb") as f:
        onnx_data = f.read()
    print(f"  ONNX size: {len(onnx_data) / 1024:.1f} KB")

    # CompileOptions — minimal config, let nncase detect from ONNX
    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.dump_ir = False
    compile_options.dump_asm = False

    if quantize:
        compile_options.quant_type = "uint8"

    # Compiler
    compiler = nncase.Compiler(compile_options)

    # Import ONNX
    print("\n[1/3] Importing ONNX...")
    import_options = nncase.ImportOptions()
    compiler.import_onnx(onnx_data, import_options)
    print("  Import OK")

    # PTQ calibration
    if quantize:
        print("[2/3] Setting up PTQ quantization...")
        ptq_options = nncase.PTQTensorOptions()
        ptq_options.samples_count = 5

        # Use NCHW calibration data matching the ONNX model format
        calib_data = np.random.rand(5, 3, 224, 224).astype(np.float32)
        ptq_options.set_tensor_data(calib_data.tobytes())
        compiler.use_ptq(ptq_options)
        print("  PTQ setup OK")

    # Compile
    print("[3/3] Compiling...")
    compiler.compile()
    print("  Compile OK")

    # Generate kmodel
    kmodel = compiler.gencode_tobytes()
    with open(str(kmodel_path), "wb") as f:
        f.write(kmodel)

    size_kb = kmodel_path.stat().st_size / 1024
    size_mb = size_kb / 1024

    print(f"\n{'='*50}")
    print(f"  SUCCESS!")
    print(f"  Size: {size_kb:.1f} KB ({size_mb:.2f} MB)")
    print(f"  Path: {kmodel_path}")
    print(f"{'='*50}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="best.onnx")
    parser.add_argument("--output", "-o", default="best.kmodel")
    parser.add_argument("--target", default="k210")
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
            quantize=not args.no_quantize,
        )
    except Exception as e:
        print(f"\nERROR: Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Size check
    size_mb = kmodel_path.stat().st_size / (1024 * 1024)
    if size_mb > 6.0:
        print(f"WARNING: kmodel {size_mb:.1f}MB exceeds 6MB limit!")
        sys.exit(1)
    else:
        print(f"PASS: {size_mb:.1f}MB within 6MB limit")


if __name__ == "__main__":
    main()
