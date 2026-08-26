"""YOLOv9-T (WongKinYiu/YOLO, MIT) -> fully INT8 TFLite via TinyNeuralNetwork PTQ.

- Cuts the graph at the raw Detection head convs (class_conv / anchor_conv), i.e.
  before Anchor2Vec (rearrange + 5D softmax + Conv3d), which cannot run on Ethos-U.
- Drops the training-only auxiliary branch from the model config.
- Post-training static quantization, asymmetric per-channel INT8, calibrated with
  a small set of COCO val2017 images using the repo's letterbox preprocessing.
- Exports TFLite with signed INT8 input/output (fuse_quant_dequant) and rewrites
  grouped convs (groups=4 in the anchor branch) into SPLIT/CONV_2D/CONCAT so that
  every operator can be mapped onto the Ethos-U55 by vela.
"""

import sys
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEPLOY_DIR.parent
YOLO_ROOT = REPO_ROOT / "YOLO"
TINYNN_ROOT = REPO_ROOT / "TinyNeuralNetwork"
WEIGHT_PATH = YOLO_ROOT / "weights" / "v9-t.pt"
CALIB_DIR = DEPLOY_DIR / "calib_images"
OUT_DIR = DEPLOY_DIR / "out"
TFLITE_PATH = OUT_DIR / "yolov9t_int8.tflite"
IMG_SIZE = 640

# use the local (patched) checkouts rather than anything pip-installed
sys.path.insert(0, str(YOLO_ROOT))
sys.path.insert(0, str(TINYNN_ROOT))

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import functional as TF

from yolo.model import module as yolo_module
from yolo.model.yolo import create_model

from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.quantizer import PostQuantizer
from tinynn.graph.tracer import model_tracer


def detection_forward_raw(self, x):
    """Detection head cut point: raw conv outputs only (NPU-friendly)."""
    return self.class_conv(x), self.anchor_conv(x)


def detection_init_ckpt_compat(self, in_channels, num_classes, *, reg_max=16, use_group=True):
    """Detection.__init__ with class_neck = num_classes (80).

    The v1.0-alpha v9-t.pt checkpoint was trained before the repo changed the
    class neck to min(num_classes * 2, 128); with the current code the class
    branch shapes mismatch and its weights silently stay random
    ("Weight Mismatch for Layer 22" warning).
    """
    nn.Module.__init__(self)
    groups = 4 if use_group else 1
    anchor_channels = 4 * reg_max
    first_neck, in_channels = in_channels
    anchor_neck = max(yolo_module.round_up(first_neck // 4, groups), anchor_channels, reg_max)
    class_neck = max(first_neck, num_classes)

    Conv = yolo_module.Conv
    self.anchor_conv = nn.Sequential(
        Conv(in_channels, anchor_neck, 3),
        Conv(anchor_neck, anchor_neck, 3, groups=groups),
        nn.Conv2d(anchor_neck, anchor_channels, 1, groups=groups),
    )
    self.class_conv = nn.Sequential(
        Conv(in_channels, class_neck, 3), Conv(class_neck, class_neck, 3), nn.Conv2d(class_neck, num_classes, 1)
    )
    self.anc2vec = yolo_module.Anchor2Vec(reg_max=reg_max)
    self.anchor_conv[-1].bias.data.fill_(1.0)
    self.class_conv[-1].bias.data.fill_(-10)


class YOLOv9Deploy(nn.Module):
    """Flattens the 3-scale Main head outputs into a 6-tensor tuple."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        main = self.model(x)["Main"]
        (c3, a3), (c4, a4), (c5, a5) = main
        return c3, a3, c4, a4, c5, a5


def build_float_model():
    cfg = OmegaConf.load(YOLO_ROOT / "yolo/config/model/v9-t.yaml")
    OmegaConf.set_struct(cfg, False)
    # training-only auxiliary branch (comes last, does not shift layer indices)
    del cfg.model["auxiliary"]
    yolo_module.Detection.__init__ = detection_init_ckpt_compat
    yolo_module.Detection.forward = detection_forward_raw
    model = create_model(cfg, weight_path=str(WEIGHT_PATH))
    deploy = YOLOv9Deploy(model)
    deploy.eval()
    return deploy


def letterbox(path, size=IMG_SIZE):
    """Match the repo's PadAndResize + to_tensor eval preprocessing."""
    image = Image.open(path).convert("RGB")
    w, h = image.size
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
    padded = Image.new("RGB", (size, size), (114, 114, 114))
    padded.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return TF.to_tensor(padded).unsqueeze(0)  # 1x3xHxW, [0,1]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)

    with model_tracer():
        model = build_float_model()

        with torch.no_grad():
            outs = model(dummy_input)
        print("Float outputs:", [tuple(o.shape) for o in outs])

        quantizer = PostQuantizer(
            model,
            dummy_input,
            work_dir=str(OUT_DIR / "quantizer"),
            config={"asymmetric": True, "per_tensor": False},
        )
        ptq_model = quantizer.quantize()

    ptq_model.eval()
    calib_files = sorted(CALIB_DIR.glob("*.jpg"))
    assert calib_files, f"no calibration images in {CALIB_DIR}"
    print(f"Calibrating on {len(calib_files)} images ...")
    with torch.no_grad():
        for i, f in enumerate(calib_files):
            ptq_model(letterbox(f))
            print(f"  [{i + 1}/{len(calib_files)}] {f.name}")

    with torch.no_grad():
        ptq_model = quantizer.convert(ptq_model)
        torch.backends.quantized.engine = quantizer.backend

        converter = TFLiteConverter(
            ptq_model,
            dummy_input,
            tflite_path=str(TFLITE_PATH),
            quantize_target_type="int8",
            fuse_quant_dequant=True,      # signed INT8 model input/output
            group_conv_rewrite=True,      # split grouped convs for Ethos-U
            rewrite_quantizable=True,
        )
        converter.convert()

    print("Saved:", TFLITE_PATH)


if __name__ == "__main__":
    main()
