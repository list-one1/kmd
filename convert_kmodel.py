#!/usr/bin/env python3
"""
YOLO ONNX → K210 kmodel converter using nncase 1.x API.

Based on pytorch-k210 proven pipeline.
Requires: nncase==1.9.0.20230322
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def convert_onnx_to_kmodel(onnx_path, kmodel_path, target="k210", quantize=True):
    """Convert ONNX model to K210 kmodel."""
    import nncase

    print(f"\n{'='*50}")
    print(f"  nncase: ONNX -> {target} kmodel")
    print(f"{'='*50}")
    print(f"  Input:     {onnx_path}")
    print(f"  Output:    {kmodel_path}")
    print(f"  Quantize:  {'INT8' if quantize else 'FP32'}")

    # Run ONNX shape inference to fill missing value info
    print("\n  Running shape inference...")
    import onnx as onnx_lib
    onnx_model = onnx_lib.load(str(onnx_path))

    try:
        onnx_model = onnx_lib.shape_inference.infer_shapes(onnx_model)
        print(f"  Shape inference OK ({len(onnx_model.graph.value_info)} value_info entries)")
    except Exception as e:
        print(f"  Shape inference failed: {e}")
        onnx_model = None

    if onnx_model is not None:
        # Try onnxsim, but don't fail if it errors
        try:
            import onnxsim
            input_shapes = {}
            input_all = [n.name for n in onnx_model.graph.input]
            input_init = [n.name for n in onnx_model.graph.initializer]
            for n in onnx_model.graph.input:
                if n.name not in input_init:
                    s = [d.dim_value if d.dim_value != 0 else 1 for d in n.type.tensor_type.shape.dim]
                    input_shapes[n.name] = s
            onnx_model, check = onnxsim.simplify(onnx_model, input_shapes=input_shapes)
            print(f"  ONNX simplify: {'OK' if check else 'validation failed'}")
        except ImportError:
            print("  onnxsim not available, using shape inference only")
        except Exception as e:
            print(f"  ONNX simplify skipped: {e}")

        onnx_data = onnx_model.SerializeToString()
        print(f"  ONNX size: {len(onnx_data) / 1024:.1f} KB")
    else:
        with open(str(onnx_path), "rb") as f:
            onnx_data = f.read()
        print(f"  ONNX size: {len(onnx_data) / 1024:.1f} KB (original)")

    # Minimal CompileOptions — nncase auto-detects from ONNX
    print("\n  Setting up CompileOptions...")
    compile_options = nncase.CompileOptions()
    compile_options.target = target
    print(f"  target={target}")

    if quantize:
        compile_options.quant_type = "uint8"
        print("  quant_type=uint8")

    # Create compiler
    print("\n[1/3] Importing ONNX...")
    compiler = nncase.Compiler(compile_options)
    import_options = nncase.ImportOptions()
    compiler.import_onnx(onnx_data, import_options)
    print("  Import OK")

    # PTQ
    if quantize:
        print("[2/3] Setting up PTQ...")
        ptq_options = nncase.PTQTensorOptions()
        ptq_options.samples_count = 5
        calib_data = np.random.rand(5, 3, 224, 224).astype(np.float32)
        ptq_options.set_tensor_data(calib_data.tobytes())
        compiler.use_ptq(ptq_options)
        print("  PTQ OK")

    # Compile
    print("[3/3] Compiling...")
    compiler.compile()
    print("  Compile OK")

    # Generate
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
    print(f"PASS: {size_mb:.1f}MB")
    if size_mb > 6.0:
        print(f"WARNING: Exceeds 6MB limit!")
        sys.exit(1)


if __name__ == "__main__":
    main()
