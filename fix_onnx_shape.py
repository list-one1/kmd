"""Fix spatial dimensions in pruned ONNX - change 1x1 conv back to 3x3."""
import numpy as np
import onnx

INPUT_PATH = "best.onnx"
OUTPUT_PATH = "best_fixed.onnx"

model = onnx.load(INPUT_PATH)

conv_fixes = {}

for node in model.graph.node:
    if node.op_type != "Conv":
        continue
    # Only fix conv_6 and conv_7 (intermediate), not conv_8 (output layer)
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
    if len(w.shape) != 4:
        continue
    out_ch, in_ch, kh, kw = w.shape
    if kh == 1 and kw == 1:
        name = node.name
        print(f"  Found 1x1 Conv: {name} weight={w.shape}")

        # Reshape to 3x3: place 1x1 weight at center, zeros elsewhere
        new_w = np.zeros((out_ch, in_ch, 3, 3), dtype=np.float32)
        new_w[:, :, 1, 1] = w[:, :, 0, 0]

        # Update initializer
        weight_init.CopyFrom(onnx.numpy_helper.from_array(new_w, weight_name))

        # Update kernel_shape attribute
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                attr.ints[:] = [3, 3]
                break

        # Set pads to SAME padding (preserves spatial dimensions)
        for attr in node.attribute:
            if attr.name == "pads":
                attr.ints[:] = [1, 1, 1, 1]
                break

        conv_fixes[name] = {
            "old_shape": list(w.shape),
            "new_shape": list(new_w.shape),
        }

# Clear stale value_info
while len(model.graph.value_info) > 0:
    del model.graph.value_info[0]

# Re-run shape inference
model = onnx.shape_inference.infer_shapes(model)

print(f"\nFixed {len(conv_fixes)} conv layers:")
for name, info in conv_fixes.items():
    print(f"  {name}: {info['old_shape']} -> {info['new_shape']}")

for o in model.graph.output:
    s = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(f"\nOutput: {o.name} shape={s}")

size_kb = len(model.SerializeToString()) / 1024
print(f"Model size: {size_kb:.1f} KB")

onnx.save(model, OUTPUT_PATH)
print(f"\nSaved: {OUTPUT_PATH}")
