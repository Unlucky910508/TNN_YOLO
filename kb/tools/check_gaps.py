"""Re-derive the "PyTorch cannot quantize it and TinyNN has no recipe" set.

`quantization_gaps.md` is hand-written, because the recommendations in it are judgement,
not data. This script re-derives the *data* half against the TinyNeuralNetwork checkout and
PyTorch version actually installed, and reports any drift from what the document claims.

Run it after bumping torch or updating the TinyNeuralNetwork checkout.
"""

import argparse
import sys
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent
TINYNN_ROOT = KB_DIR.parent / "TinyNeuralNetwork"

# The residual set quantization_gaps.md is written against. Keep in sync with that document.
DOCUMENTED_RESIDUAL = {
    'atan', 'atan2', 'cos', 'exp', 'group_norm', 'hardsigmoid', 'instance_norm', 'layer_norm',
    'log', 'mm', 'norm', 'pow', 'rsqrt', 'sin', 'std', 'var',
    'torch.nn.GroupNorm', 'torch.nn.InstanceNorm1d', 'torch.nn.InstanceNorm2d', 'torch.nn.RNN',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tinynn-root", default=str(TINYNN_ROOT))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.tinynn_root).resolve()))

    import torch
    from packaging.version import Version

    from tinynn.converter.operators.torch import OPERATOR_CONVERTER_DICT
    from tinynn.graph.quantization.quantizer import (
        FUNCTIONAL_MODULE_MAPPING,
        KNOWN_QSTATS,
        Q_MODULES_MAPPING,
        REWRITE_QUANTIZABLE_RULE_LIST,
        UNSUPPORTED_PYTORCH_QUANTIZATION_OP_LIST as UNSUPPORTED,
    )
    from tinynn.graph.tracer import qualified_name

    def name(key):
        return key if isinstance(key, str) else qualified_name(key, short=True)

    torch_version = torch.__version__.split('+')[0]

    # An entry with a version string is only unsupported *below* that version. See the guard in
    # QATQuantizer._rewrite_quantize_graph: `supported_version is None or torch < supported_version`.
    lifted = {
        name(k): v
        for k, v in UNSUPPORTED.items()
        if v is not None and Version(torch_version) >= Version(v)
    }

    # Table 1 of docs/quantization_support.md
    table1 = set(map(name, UNSUPPORTED)) | set(map(name, Q_MODULES_MAPPING))

    # Table 2 ("extra flags"), mirroring scripts/gen_quantized_docs.py
    functional_ops = {(k,) for k in UNSUPPORTED.keys() & FUNCTIONAL_MODULE_MAPPING.keys()}
    covered = (
        {(k,) for k in KNOWN_QSTATS}
        | REWRITE_QUANTIZABLE_RULE_LIST
        | {(k,) for k in Q_MODULES_MAPPING}
        | functional_ops
    )
    table2 = {name(k) for group in covered for k in group}

    residual = table1 - table2 - set(lifted)

    print(f"torch                 {torch.__version__}")
    print(f"table1 (no PT quant)  {len(table1)}")
    print(f"table2 (TNN recipe)   {len(table2)}")
    print(f"version-lifted        {len(lifted)}  {sorted(lifted)}")
    print(f"residual              {len(residual)}")

    # Which residual ops still reach TFLite (as a float island) vs fail outright.
    aten_names = {
        'torch.nn.RNN': ('aten::rnn_tanh', 'aten::rnn_relu'),
        'torch.nn.GroupNorm': ('aten::group_norm',),
        'torch.nn.InstanceNorm1d': ('aten::instance_norm',),
        'torch.nn.InstanceNorm2d': ('aten::instance_norm',),
    }
    print("\nresidual op -> converter registration")
    hard = []
    for op in sorted(residual):
        keys = aten_names.get(op, (f'aten::{op}',))
        ok = [k for k in keys if k in OPERATOR_CONVERTER_DICT]
        status = 'float island' if ok else 'HARD FAIL (no converter)'
        if not ok:
            hard.append(op)
        print(f"  {op:26s} {status}")
    print(f"\nhard failures: {sorted(hard)}")

    added = residual - DOCUMENTED_RESIDUAL
    removed = DOCUMENTED_RESIDUAL - residual
    if added or removed:
        print("\n!! DRIFT from quantization_gaps.md")
        if added:
            print(f"   newly residual (document them): {sorted(added)}")
        if removed:
            print(f"   no longer residual (drop them): {sorted(removed)}")
        return 1
    print("\nMatches quantization_gaps.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
