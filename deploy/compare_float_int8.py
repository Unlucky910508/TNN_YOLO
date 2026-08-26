"""Sanity check: cosine similarity between float PyTorch outputs and dequantized
INT8 TFLite outputs on a real image (not part of proving int8-ness; just ensures
calibration wasn't degenerate)."""

import sys
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY_DIR))

import numpy as np
import torch

from ptq_convert import build_float_model, letterbox

IMG = str(DEPLOY_DIR / "calib_images" / "000000000139.jpg")
TFLITE = str(DEPLOY_DIR / "out" / "yolov9t_int8.tflite")

x = letterbox(IMG)

model = build_float_model()
with torch.no_grad():
    ref = [o.numpy() for o in model(x)]

import tensorflow as tf

interp = tf.lite.Interpreter(
    model_path=TFLITE,
    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
scale, zp = inp["quantization"]
xq = np.clip(np.round(x.numpy() / scale + zp), -128, 127).astype(np.int8)
xq = np.transpose(xq, (0, 2, 3, 1))  # NCHW -> NHWC
interp.set_tensor(inp["index"], xq)
interp.invoke()

outs = []
for d in interp.get_output_details():
    q = interp.get_tensor(d["index"]).astype(np.float32)
    s, z = d["quantization"]
    outs.append(np.transpose((q - z) * s, (0, 3, 1, 2)))  # NHWC -> NCHW

# match by shape (output order may differ)
used = set()
for i, r in enumerate(ref):
    for j, o in enumerate(outs):
        if j in used or o.shape != r.shape:
            continue
        a, b = r.flatten(), o.flatten()
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        mae = float(np.mean(np.abs(a - b)))
        if cos > 0.5:  # assume matched
            used.add(j)
            kind = "class" if r.shape[1] == 80 else "anchor"
            print(f"ref[{i}] ({kind}, {r.shape}): cosine={cos:.4f}  MAE={mae:.4f}  "
                  f"float_range=[{a.min():.2f},{a.max():.2f}]")
            break
    else:
        print(f"ref[{i}] {r.shape}: NO MATCH FOUND")
