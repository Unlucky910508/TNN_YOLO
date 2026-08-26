# TNN_YOLO

YOLOv9-T (PyTorch) → fully-INT8 TFLite → Ethos-U55 vela, using
[TinyNeuralNetwork](https://github.com/alibaba/TinyNeuralNetwork) post-training quantization.

Result of the current pipeline (`deploy/vela_run3.log`):

```
Accelerator configuration                Ethos_U55_64
CPU operators = 0 (0.0%)
NPU operators = 801 (100.0%)
Total SRAM used                               3205.09 KiB
Total Off-chip Flash used                     2151.70 KiB
```

## Layout

| Path | What |
|---|---|
| `deploy/ptq_convert.py` | The pipeline: build YOLOv9-T, cut the head before `Anchor2Vec`, PTQ-calibrate, export signed-INT8 TFLite. |
| `deploy/verify_tflite.py` | Asserts the export is fully signed-INT8 (input, output, weights, activations) and lists the operator set. |
| `deploy/compare_float_int8.py` | Cosine similarity between float PyTorch and dequantized INT8 outputs — catches degenerate calibration. |
| `convert_yolov9_int8_tflite.py` | Earlier, simpler conversion script (dummy calibration). Superseded by `deploy/ptq_convert.py`. |
| `kb/` | Knowledge base for driving TinyNN conversions: operator quantization gaps, an indexed error catalogue, and the scripts that regenerate both. |
| `patches/tinynn-local.patch` | **Required.** Two fixes to TinyNeuralNetwork the pipeline depends on (see below). |
| `TinyNeuralNetwork/`, `YOLO/` | Upstream repos, as submodules pinned to the commits this was built against. |

Weights, datasets, calibration images and build outputs are not tracked — see
[`.gitignore`](.gitignore).

## Setup

```bash
git clone --recurse-submodules https://github.com/Unlucky910508/TNN_YOLO.git
cd TNN_YOLO

# the two local TinyNN fixes the pipeline depends on
git -C TinyNeuralNetwork apply ../patches/tinynn-local.patch

pip install -r TinyNeuralNetwork/requirements.txt -r YOLO/requirements.txt
```

Then fetch what is not in the repo:

- `YOLO/weights/v9-t.pt` — YOLOv9-T weights, from the
  [YOLO](https://github.com/MultimediaTechLab/YOLO) release assets.
- `deploy/calib_images/` — ~34 COCO val2017 JPEGs for PTQ calibration.

## Run

```bash
python deploy/ptq_convert.py          # -> deploy/out/yolov9t_int8.tflite
python deploy/verify_tflite.py        # fully-INT8 check + op histogram
python deploy/compare_float_int8.py   # float vs INT8 cosine similarity

vela --accelerator-config ethos-u55-64 \
     --output-dir deploy/out/vela \
     deploy/out/yolov9t_int8.tflite
```

## The two TinyNN patches

`patches/tinynn-local.patch` (against `TinyNeuralNetwork@cddc412`):

1. **`converter/operators/optimize.py`** — when a grouped conv is rewritten into
   `SPLIT`/`CONV_2D`/`CONCAT`, the per-channel quantization params of the weight and bias were
   copied whole to every chunk instead of being sliced along with the tensor. The anchor branch
   uses `groups=4`, so without this the split convs carry the wrong scales.
2. **`graph/quantization/quantizer.py`** — unify the quantization params across the inputs of a
   `cat` fed by QSiLU. XNNPACK rejects `CONCATENATION` whose inputs disagree on qparams, and vela
   would otherwise have to insert requantization.

Both are candidates to send upstream; until then they live here as a patch so the submodule can
stay pinned to plain upstream.

## Why the head is cut

`Anchor2Vec` (rearrange + 5-D softmax + `Conv3d`) has no Ethos-U mapping, so `ptq_convert.py`
cuts the graph at the raw detection-head convs (`class_conv` / `anchor_conv`) and leaves the
DFL decode to the host. The training-only auxiliary branch is dropped from the model config.
