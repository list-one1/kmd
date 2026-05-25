# KMD Project Context — K210 / MaixCAM 模型转换

## 项目目标

将 YOLOv2 扑克牌检测模型（53类）转换为 K210 和 MaixCAM 可用格式，部署到 GitHub。

## 仓库：https://github.com/list-one1/kmd

---

## 已完成的文件

### 核心文件

| 文件 | 用途 |
|------|------|
| `best.pt` | 训练好的 YOLOv2 模型（原始权重） |
| `best.onnx` | 导出的 ONNX（7.3 MB, FP32, 215×215 输入） |
| `best.kmodel` | K210 转换产物（1.83 MB, INT8 量化） 已提交 git |
| `convert_kmodel.py` | K210 转换脚本（nncase Python API） |
| `fix_onnx_shape.py` | 修复剪枝后 ONNX 的空间维度问题 |
| `export_k210_onnx.py` | YOLO .pt → ONNX 导出脚本 |

### CI

| 文件 | 用途 |
|------|------|
| `.github/workflows/kmodel.yml` | K210 kmodel 验证（检查文件存在 + < 6MB） |

### 技能

| 文件 | 用途 |
|------|------|
| `skills/k210-kmodel-conversion.md` | K210 转换完整流程技能 |
| `skills/maixcam-cvimodel-conversion.md` | MaixCAM cvimodel 转换技能 |

---

## K210 转换流水线

### 关键约束

- **芯片**: K210 (Kendryte)
- **格式**: .kmodel
- **输入限制**: ≤ 224×224（KPU 只有 2 MB 内存）
- **总大小**: < 6 MB
- **量化**: INT8（uint8）
- **工具链**: nncase Python API（非 ncc CLI）

### nncase 版本问题（重要！）

```
nncase Python API 版本树：
├── v0.x (ncc CLI) — 官方支持 K210，输入 TFLite
│   └── v0.2.0-beta4 是最后一版确认可用
├── v1.x (Python API) — 我们用的 v1.9.0.20230322
│   └── 转换流程能跑通，生成 1.83 MB 文件
│   └── ⚠️ 未在 K210 实机上验证！
├── v2.x — 砍掉 K210，只支持 K230+
```

**当前状态**: 使用了 nncase 1.9.0.20230322 转换，文件生成成功，CI 通过，但**未在 K210 实机验证**。如果有问题，需要降级到 ncc CLI 方案：`ONNX → TFLite → ncc → kmodel`。

### 转换流程

```
[1] best.pt → best.onnx (ultralytics export)
    - imgsz=224, opset=11, simplify=True
    - 此时 ONNX 被自动剪枝，conv_6/conv_7 从 3×3 变成 1×1
    
[2] best.onnx → 修复空间维度 (fix_onnx_shape.py)
    - 把 conv_6/conv_7 的 1×1 核补成 3×3（权重放中心、周围填零）
    - 设置 pads=[1,1,1,1]（SAME padding，保持空间维度）
    - 清除 value_info 重新 shape_inference
    - 数学等价，零精度损失
    
[3] 修复后的 ONNX → nncase 转换 (convert_kmodel.py)
    - onnx.shape_inference.infer_shapes() 补充 value_info
    - 可选 onnxsim simplify
    - nncase.CompileOptions(target="k210", quant_type="uint8")
    - PTQ: 5 个随机校准样本 (3,224,224)
    - compiler.compile() → gencode_tobytes()
    
[4] 验证: kmodel < 6MB
    - 当前: 1876 KB (1.83 MB) ✓
```

### nncase 1.x API 要点

```python
import nncase

# 关键：不设置 input_shape / input_type / input_layout / output_layout
# nncase 从 ONNX 自动检测
compile_options = nncase.CompileOptions()
compile_options.target = "k210"
compile_options.quant_type = "uint8"  # INT8

compiler = nncase.Compiler(compile_options)
compiler.import_onnx(onnx_data, nncase.ImportOptions())

# INT8 PTQ 量化
ptq_options = nncase.PTQTensorOptions()
ptq_options.samples_count = 5
calib_data = np.random.rand(5, 3, 224, 224).astype(np.float32)
ptq_options.set_tensor_data(calib_data.tobytes())
compiler.use_ptq(ptq_options)

compiler.compile()
kmodel = compiler.gencode_tobytes()
```

### 常见错误速查（K210）

| 错误 | 原因 | 修复 |
|------|------|------|
| `Can't find value info for /conv/Conv_output_0` | ONNX 缺少中间 tensor shape | 运行 shape_inference.infer_shapes() |
| `Shape mismatch: [1,290,11,11] -> [1,290,7,7]` | 剪枝后 1×1 卷积破坏了空间维度链 | fix_onnx_shape.py 修复 |
| `existing shape differ in dimension 2: (3) vs (7)` | value_info 缓存了旧 shape | 修复权重后清除 value_info 再 re-infer |
| CI 中 `python` 找不到 | Linux CI 只有 python3 | 所有命令用 python3 |
| `nncase.__version__` 不存在 | nncase 1.x 没有这个属性 | 不要引用 __version__ |

---

## MaixCAM 转换流水线

### 关键约束

- **芯片**: cv181x
- **格式**: .cvimodel + .mud
- **输入**: 640×640
- **量化**: BF16（默认）或 INT8
- **工具链**: Sophgo TPU-MLIR（Docker: sophgo/tpuc_dev:latest）

### 转换流程

```
[1] best.pt → best.onnx (640×640, opset 12)
[2] 生成 .mud 配置文件（类名、预处理参数）
[3] Docker 内:
    a) ONNX → MLIR (model_transform.py)
    b) INT8 校准 (run_calibration.py，可选)
    c) MLIR → .cvimodel (model_deploy.py)
[4] 部署: .cvimodel + .mud → MaixCAM 设备
```

### .mud 配置要点

```ini
[basic]
type = cvimodel
model = model_bf16.cvimodel

[extra]
model_type = yolov8          # 也适用于 YOLOv11/v12
input_type = rgb
mean = 0, 0, 0
scale = 0.00392156862745098   # = 1/255
labels = 类名,逗号,分隔
```

### 相关仓库

- `C:\Users\1\Desktop\2\maixcam-convert\` — MaixCAM 完整转换项目（含 Docker、CI）

---

## GitHub Actions CI

### K210 当前状态

- 工作流: `K210 kmodel verification`
- 触发: push 到 best.onnx / best.kmodel / workflow 文件
- 动作: 检查 best.kmodel 存在且 < 6 MB，上传 artifact
- 状态: 绿色 ✓

### Git 推送注意事项

- 本机有 SOCKS5 代理 (127.0.0.1:1080)，HTTPS git push 不兼容
- 推送用: `git -c http.proxy= -c https.proxy= push origin main`

---

## 待验证 / 待完成

1. **K210 实机验证** — nncase 1.9.0.20230322 生成的 kmodel 能否在 K210 上运行？
2. **检测精度测试** — 224×224 + INT8 量化后的实际检测效果？
3. **ncc 备选方案** — 如果 Python API 不可用，切换到 `ONNX → TFLite → ncc → kmodel`
4. **MaixCAM 转换** — 实际跑通一次 Docker 转换

---

## 技能文件路径

- Claude Code 技能: `C:\Users\1\.claude\skills\k210-kmodel-conversion\SKILL.md`
- Claude Code 技能: `C:\Users\1\.claude\skills\maixcam-cvimodel-conversion\SKILL.md`
- Git 副本: `skills/k210-kmodel-conversion.md` 和 `skills/maixcam-cvimodel-conversion.md`
