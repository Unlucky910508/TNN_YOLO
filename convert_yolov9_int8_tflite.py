"""Convert YOLOv9 (WongKinYiu/YOLO, PyTorch) to INT8 TFLite via TinyNeuralNetwork.

Pipeline: build model -> wrap to flat-tensor outputs -> PTQ (dummy calibration)
-> torch quantized model -> TFLite (full INT8).

Run with the `yolo` conda env:
    /home/nick/miniconda3/envs/yolo/bin/python convert_yolov9_int8_tflite.py
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, 'TinyNeuralNetwork'))
sys.path.insert(0, os.path.join(ROOT, 'YOLO'))

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from yolo.model.yolo import create_model
from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.quantizer import PostQuantizer
from tinynn.graph.tracer import model_tracer


class DeployYOLO(nn.Module):
    """Wraps the YOLO model into a flat-tensor-output deploy model.

    Only the "Main" branch is executed (the AUX branch is training-only).
    Each detection scale yields (class_x, anchor_x, vector_x); anchor_x is the
    raw DFL distribution used only by the training loss, so we export
    class_x (B, num_classes, h, w) and vector_x (B, 4, h, w) per scale.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x, shortcut="Main")["Main"]
        flat = []
        for class_x, _anchor_x, vector_x in outputs:
            flat.append(class_x)
            flat.append(vector_x)
        return tuple(flat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='v9-c', help='model config name, e.g. v9-c / v9-s / v9-t')
    parser.add_argument('--size', type=int, default=640, help='input size (square)')
    parser.add_argument('--weights', default=None, help='path to .pt weights (default: weights/<model>.pt)')
    parser.add_argument('--output', default=None, help='output tflite path')
    parser.add_argument('--calib-iters', type=int, default=8, help='dummy calibration iterations')
    args = parser.parse_args()

    weight_path = args.weights or os.path.join(ROOT, 'weights', f'{args.model}.pt')
    tflite_path = args.output or os.path.join(ROOT, 'out', f'yolo{args.model.replace("-", "")}_int8.tflite')
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    work_dir = os.path.join(ROOT, 'out', 'ptq')

    cfg_path = os.path.join(ROOT, 'YOLO', 'yolo', 'config', 'model', f'{args.model}.yaml')
    model_cfg = OmegaConf.load(cfg_path)

    dummy_input = torch.rand((1, 3, args.size, args.size))

    with model_tracer():
        model = create_model(model_cfg, weight_path=weight_path, class_num=80)
        model = DeployYOLO(model)
        model.eval()

        # int8 asymmetric per-channel matches the current TFLite quantization spec
        quantizer = PostQuantizer(
            model, dummy_input, work_dir=work_dir,
            config={'asymmetric': True, 'per_tensor': False},
        )
        ptq_model = quantizer.quantize()

    # Dummy calibration: random inputs only populate the observers so the
    # conversion can proceed; ranges are not representative of real accuracy.
    ptq_model.eval()
    with torch.no_grad():
        for i in range(args.calib_iters):
            ptq_model(torch.rand((1, 3, args.size, args.size)))
            print(f'calibration {i + 1}/{args.calib_iters}')

    with torch.no_grad():
        ptq_model = quantizer.convert(ptq_model)
        torch.backends.quantized.engine = quantizer.backend

        converter = TFLiteConverter(
            ptq_model, dummy_input,
            tflite_path=tflite_path,
            quantize_target_type='int8',
            fuse_quant_dequant=True,
            rewrite_quantizable=True,
        )
        converter.convert()

    print(f'\nDone: {tflite_path}')


if __name__ == '__main__':
    main()
