---
name: maixcam-cvimodel-conversion
description: Convert YOLO ONNX models to MaixCAM cvimodel format using Sophgo TPU-MLIR for cv181x deployment
---

# MaixCAM cvimodel Conversion

Convert YOLO ONNX models to MaixCAM `.cvimodel` + `.mud` format using Sophgo TPU-MLIR toolchain, targeting the cv181x processor.

## Pipeline Overview

```
YOLO .pt → ONNX (640×640, opset 12) → MLIR → Calibration (INT8 only) → .cvimodel + .mud
```

Two quantization options:
- **BF16** (default): no calibration images needed, good accuracy
- **INT8**: requires calibration images, smaller/faster model

## Prerequisites

Docker with `sophgo/tpuc_dev:latest` image, or install `tpu_mlir` directly on Linux.

```bash
docker pull sophgo/tpuc_dev:latest
pip install ultralytics torch onnx  # for ONNX export
```

## Step 1: Export ONNX

Export YOLO model at 640×640 with opset 12:

```python
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='onnx', imgsz=640, opset=12, simplify=True)
```

## Step 2: Generate .mud Config

The `.mud` file tells MaixCAM how to use the model:

```ini
[basic]
type = cvimodel
model = mymodel_bf16.cvimodel

[extra]
model_type = yolov8
input_type = rgb
mean = 0, 0, 0
scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098
labels = Ace,2,3,4,5,6,7,8,9,10,Jack,Queen,King,...
```

Key fields:
- `model_type`: `yolov8` for YOLOv8/v11/v12
- `scale`: `0.00392156862745098` = 1/255 (normalize uint8 to [0,1])
- `mean`: `0,0,0` (no mean subtraction)
- `labels`: comma-separated class names

## Step 3: Convert via Docker

### BF16 (no calibration needed)

```bash
MODEL_NAME="yolo11n_card"
INPUT_SIZE=640

# ONNX → MLIR
model_transform.py \
  --model_name "${MODEL_NAME}" \
  --model_def best.onnx \
  --input_shapes "[[1,3,${INPUT_SIZE},${INPUT_SIZE}]]" \
  --mean "0,0,0" \
  --scale "0.00392156862745098,0.00392156862745098,0.00392156862745098" \
  --pixel_format rgb \
  --channel_format nchw \
  --output_names "output0" \
  --tolerance 0.99,0.99 \
  --mlir "workspace/${MODEL_NAME}.mlir"

# MLIR → cvimodel
model_deploy.py \
  --mlir "workspace/${MODEL_NAME}.mlir" \
  --quantize BF16 \
  --quant_input \
  --processor cv181x \
  --tolerance 0.99,0.99 \
  --model "workspace/${MODEL_NAME}_bf16.cvimodel"
```

### INT8 (needs calibration images)

```bash
# ONNX → MLIR (same as above)
model_transform.py ...

# Calibration
run_calibration.py "workspace/${MODEL_NAME}.mlir" \
  --dataset calibration_images \
  --input_num 100 \
  -o "workspace/${MODEL_NAME}_cali_table"

# MLIR → cvimodel (INT8)
model_deploy.py \
  --mlir "workspace/${MODEL_NAME}.mlir" \
  --quantize INT8 \
  --quant_input \
  --calibration_table "workspace/${MODEL_NAME}_cali_table" \
  --processor cv181x \
  --tolerance 0.9,0.6 \
  --model "workspace/${MODEL_NAME}_int8.cvimodel"
```

## Step 4: Deploy to MaixCAM

Copy `.cvimodel` and `.mud` to the device, then load in MicroPython:

```python
from maix import camera, display, nn

model = nn.NN('/path/to/model.mud')

while True:
    img = camera.capture()
    results = model.forward(img)
    # process results...
```

## CI Integration

Use `sophgo/tpuc_dev:latest` as the container image:

```yaml
name: Convert ONNX to cvimodel
on:
  push:
    paths: ["best.onnx", ".github/workflows/convert.yml"]
  workflow_dispatch:
    inputs:
      quantize:
        description: "Quantization"
        default: "BF16"
        options: ["BF16", "INT8"]

jobs:
  convert:
    runs-on: ubuntu-latest
    container:
      image: sophgo/tpuc_dev:latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install tpu_mlir onnx onnxruntime --quiet

      - name: ONNX → MLIR
        run: |
          mkdir -p workspace
          model_transform.py \
            --model_name model \
            --model_def best.onnx \
            --input_shapes "[[1,3,640,640]]" \
            --mean "0,0,0" \
            --scale "0.00392156862745098,0.00392156862745098,0.00392156862745098" \
            --pixel_format rgb \
            --channel_format nchw \
            --tolerance 0.99,0.99 \
            --mlir workspace/model.mlir

      - name: MLIR → cvimodel
        run: |
          QUANTIZE="${{ inputs.quantize || 'BF16' }}"
          SUFFIX=$(echo "$QUANTIZE" | tr '[:upper:]' '[:lower:]')
          model_deploy.py \
            --mlir workspace/model.mlir \
            --quantize "${QUANTIZE}" \
            --quant_input \
            --processor cv181x \
            --tolerance 0.99,0.99 \
            --model "workspace/model_${SUFFIX}.cvimodel"

      - uses: actions/upload-artifact@v4
        with:
          name: maixcam-model
          path: workspace/*.cvimodel
          retention-days: 90
```

## Critical Rules

1. **Input size: 640×640** — MaixCAM uses larger input than K210 (224), adjust based on your model
2. **Scale is 1/255** — uint8 image normalization: pixel_value × 0.00392156862745098 = pixel/255
3. **Mean is 0,0,0** — no mean subtraction for standard YOLO preprocessing
4. **pixel_format must be rgb** — not bgr
5. **channel_format nchw** — batch, channel, height, width
6. **output_names: "output0"** — YOLOv11/v12 single output; YOLOv8 uses dual output (`/model.22/dfl/conv/Conv_output_0`, `/model.22/Sigmoid_output_0`)
7. **Processor: cv181x** — the chip in MaixCAM
8. **BF16 tolerance: 0.99,0.99** — loose tolerance since BF16 has good fidelity
9. **INT8 tolerance: 0.9,0.6** — tighter for cosine similarity, looser for euclidean
10. **.mud + .cvimodel must be paired** — both files needed on the device

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `model_transform.py: command not found` | tpu_mlir not installed | `pip install tpu_mlir` or use Docker |
| `No module named 'tpu_mlir'` | Python package missing | Install in sophgo/tpuc_dev container |
| Calibration fails | No images in calibration_images/ | Provide JPEG/PNG images, or use BF16 |
| `output_names` mismatch | Wrong output node name | Check with `onnx.load` and inspect `.graph.output` |
| INT8 accuracy drop | Poor calibration data | Use representative calibration images |
| Docker permission denied | Docker daemon not running | Start Docker Desktop or use Linux |
