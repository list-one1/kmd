---
name: k210-kmodel-conversion
description: Convert ONNX models (YOLOv2, etc.) to K210 kmodel format using nncase v1.x for K210 KPU deployment
---

# K210 kmodel Conversion

Convert ONNX models to K210 kmodel format with INT8 quantization, targeting the K210 KPU (< 6 MB).

## Prerequisites

```bash
pip install nncase==1.9.0.20230322 onnxsim onnx onnxruntime numpy
```

**nncase v1.x is required.** v2.x dropped K210 support. The API has no `__version__` attribute.

## Step 1: Export ONNX for K210

For YOLOv2 with ultralytics:

```python
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='onnx', imgsz=224, opset=11, simplify=True, dynamic=False, half=False, int8=False)
```

Key constraints: 224×224 input (K210 KPU 2 MB memory), opset 11 (nncase compatible), no dynamic shapes.

## Step 2: Inspect ONNX

Always check the model before conversion:

```python
import onnx
m = onnx.load('best.onnx')
for i in m.graph.input:
    s = [d.dim_value for d in i.type.tensor_type.shape.dim]
    print(f"Input: {i.name} shape={s}")
for o in m.graph.output:
    s = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(f"Output: {o.name} shape={s}")
```

For YOLOv2: input `[1, 3, 224, 224]`, output `[1, 290, 7, 7]` (290 = 5 anchors × (53 classes + 5 bbox)).

## Step 3: Fix Pruned Models

If the model was pruned, 3×3 convolutions may have been reduced to 1×1 kernels. This breaks spatial dimensions — nncase will complain `Shape mismatch: [1,290,11,11] -> [1,290,7,7]`.

Use this pattern to restore 3×3 kernels with SAME padding:

```python
import numpy as np
import onnx

model = onnx.load("best.onnx")

for node in model.graph.node:
    if node.op_type != "Conv":
        continue
    # Skip output layer (conv_8 in YOLOv2) — only fix intermediates
    if "conv_8" in node.name:
        continue

    weight_name = node.input[1]
    weight_init = None
    for init in model.graph.initializer:
        if init.name == weight_name:
            weight_init = init
            break
    if weight_init is None:
        continue

    w = onnx.numpy_helper.to_array(weight_init)
    out_ch, in_ch, kh, kw = w.shape
    if kh == 1 and kw == 1:
        # Place 1x1 weight at center of 3x3, zeros elsewhere
        new_w = np.zeros((out_ch, in_ch, 3, 3), dtype=np.float32)
        new_w[:, :, 1, 1] = w[:, :, 0, 0]
        weight_init.CopyFrom(onnx.numpy_helper.from_array(new_w, weight_name))

        # kernel_shape = [3, 3], pads = [1, 1, 1, 1] (SAME)
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                attr.ints[:] = [3, 3]
            elif attr.name == "pads":
                attr.ints[:] = [1, 1, 1, 1]

# Clear stale value_info and re-infer shapes
while len(model.graph.value_info) > 0:
    del model.graph.value_info[0]
model = onnx.shape_inference.infer_shapes(model)
onnx.save(model, "best_fixed.onnx")
```

## Step 4: Convert to kmodel

```python
import nncase
import onnx as onnx_lib
import numpy as np

# Load and run shape inference (nncase needs value_info)
onnx_model = onnx_lib.load("best.onnx")
onnx_model = onnx_lib.shape_inference.infer_shapes(onnx_model)

# Optional: onnxsim simplify
try:
    import onnxsim
    input_shapes = {}
    input_init = [n.name for n in onnx_model.graph.initializer]
    for n in onnx_model.graph.input:
        if n.name not in input_init:
            s = [d.dim_value if d.dim_value != 0 else 1 for d in n.type.tensor_type.shape.dim]
            input_shapes[n.name] = s
    onnx_model, _ = onnxsim.simplify(onnx_model, input_shapes=input_shapes)
except Exception:
    pass  # onnxsim is optional

onnx_data = onnx_model.SerializeToString()

# CompileOptions — do NOT set input_shape/input_type/input_layout/output_layout
compile_options = nncase.CompileOptions()
compile_options.target = "k210"
compile_options.quant_type = "uint8"  # INT8 quantization

# Import
compiler = nncase.Compiler(compile_options)
compiler.import_onnx(onnx_data, nncase.ImportOptions())

# PTQ with random calibration data (5 samples, 224x224)
ptq_options = nncase.PTQTensorOptions()
ptq_options.samples_count = 5
calib = np.random.rand(5, 3, 224, 224).astype(np.float32)
ptq_options.set_tensor_data(calib.tobytes())
compiler.use_ptq(ptq_options)

# Compile and save
compiler.compile()
kmodel = compiler.gencode_tobytes()
with open("best.kmodel", "wb") as f:
    f.write(kmodel)
```

## Step 5: Verify

```bash
stat -c%s best.kmodel  # Must be < 6291456 bytes (6 MB)
```

For YOLOv2-224: expect 1.5–2.5 MB with INT8 quantization.

## Critical Rules

1. **Never set** `input_shape`, `input_type`, `input_layout`, `output_layout` in CompileOptions — nncase auto-detects from ONNX
2. **Always run** `onnx.shape_inference.infer_shapes()` before nncase import — missing value_info causes `Can't find value info` errors
3. **Use nncase 1.9.0.20230322** — v2.x dropped K210 support and has a completely different API
4. **SAME padding** for spatial-preserving convolutions — `pads = [1, 1, 1, 1]` for kernel_size=3
5. **K210 limit**: total kmodel < 6 MB; input must be ≤ 224×224
6. **opset 11** — newer opsets may not work with nncase
7. **Clear value_info** before re-running shape inference after modifying weights — stale entries cause `shape differ` conflicts

## CI Integration

Commit the generated kmodel to the repo and use CI to verify it:

```yaml
name: K210 kmodel verification
on:
  push:
    paths: ["best.onnx", "best.kmodel", ".github/workflows/kmodel.yml"]
jobs:
  verify:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - run: |
          SIZE=$(stat -c%s best.kmodel)
          SIZE_MB=$(echo "scale=2; $SIZE / 1048576" | bc)
          [ $(echo "$SIZE_MB > 6.0" | bc) -eq 1 ] && exit 1
          echo "PASS: $SIZE_MB MB"
```

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Can't find value info for /conv/Conv_output_0` | Missing shape info in ONNX | Run `onnx.shape_inference.infer_shapes()` |
| `Shape mismatch: [1,290,11,11] -> [1,290,7,7]` | Pruned 1×1 convs lost spatial dims | Restore 3×3 kernels with SAME padding |
| `existing shape differ in dimension 2: (3) vs (7)` | Stale value_info after weight fix | Clear value_info before re-inferring |
| kmodel > 6 MB | Model too large or FP32 | Use INT8 quantization, reduce input size |
| nncase import error on Linux | Missing system libs | Use `ubuntu-22.04` base image |
