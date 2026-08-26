"""Verify the exported TFLite model is fully signed-INT8 (input, output, weights,
activations) and list the operator set."""

from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

PATH = str(Path(__file__).resolve().parent / "out" / "yolov9t_int8.tflite")

# Use builtin reference kernels without the XNNPACK delegate (XNNPACK rejects
# CONCATENATION with per-input quantization params; reference kernels and vela
# requantize instead)
interp = tf.lite.Interpreter(
    model_path=PATH,
    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
)
interp.allocate_tensors()

print("== Inputs ==")
for d in interp.get_input_details():
    print(f"  {d['name']}: {d['dtype'].__name__} {d['shape']} q={d['quantization']}")
print("== Outputs ==")
for d in interp.get_output_details():
    print(f"  {d['name']}: {d['dtype'].__name__} {d['shape']} q={d['quantization']}")

dtypes = Counter()
bad = []
for t in interp.get_tensor_details():
    name = t["name"]
    if not name:  # intermediate scratch
        continue
    dt = np.dtype(t["dtype"]).name
    dtypes[dt] += 1
    if dt not in ("int8", "int32"):  # int32 = bias, everything else must be int8
        bad.append((name, dt, t["shape"]))

print("\n== Tensor dtype histogram ==")
for k, v in sorted(dtypes.items()):
    print(f"  {k}: {v}")
if bad:
    print("\n!! Non-int8/int32 tensors:")
    for b in bad:
        print("  ", b)
else:
    print("\nAll named tensors are int8 (weights/activations) or int32 (bias). OK")

print("\n== Operator histogram ==")
try:
    from tensorflow.lite.python import analyzer

    txt = analyzer.ModelAnalyzer.analyze(model_path=PATH)
except Exception as e:
    txt = f"analyzer unavailable: {e}"

ops = Counter()
for line in str(txt).splitlines():
    line = line.strip()
    if line.startswith("Op#"):
        ops[line.split()[1].split("(")[0]] += 1
for k, v in sorted(ops.items()):
    print(f"  {k}: {v}")

# smoke inference
inp = interp.get_input_details()[0]
x = np.random.randint(-128, 128, size=inp["shape"], dtype=np.int8)
interp.set_tensor(inp["index"], x)
interp.invoke()
print("\nSmoke inference OK; output sample:",
      interp.get_tensor(interp.get_output_details()[0]["index"]).flatten()[:8])
