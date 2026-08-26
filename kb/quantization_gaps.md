# Quantization gaps: ops PyTorch will not quantize and TinyNN has no recipe for

**Scope.** `TinyNeuralNetwork/docs/quantization_support.md` lists the ops PyTorch cannot
statically quantize, then lists the subset TinyNN can still get into a quantized TFLite via extra
flags. This document covers **what is left over** — and what to do about each one.

Use it *before* converting, to decide whether a model can reach a fully-quantized graph and what
has to change if it cannot. For diagnosing an error that has already been raised, use
[`error_index.md`](error_index.md) instead.

| | |
|---|---|
| TinyNeuralNetwork | `cddc412` (+ two local patches, see [`../deploy/`](../deploy/)) |
| PyTorch | 2.5.1 |
| Verify / refresh | `python tools/check_gaps.py` |

---

## 1. First: the version trap

The second column of table 1 in `quantization_support.md` is `Minimum Supported PyTorch Version`.
The gating code is:

```python
supported_version = disable_quantize_op_list.get(cur_module.kind, torch.__version__)
return supported_version is None or LooseVersion(torch.__version__) < supported_version
```

So a row **with** a version string means *"unsupported below this version"* — it is fine on
anything newer. Only rows whose value is `None` (rendered as `/` in the generated table) are
unconditionally unsupported.

On PyTorch 2.5.1 these **8 rows are already lifted** and quantize normally, despite appearing in
the "unsupported" table:

`pad` · `torch.nn.ConstantPad1d` · `torch.nn.ConstantPad2d` · `torch.nn.ConstantPad3d` ·
`torch.nn.ZeroPad2d` · `torch.nn.ConvTranspose2d` · `torch.nn.LSTM` · `torch.nn.GRU`

`quantized::conv_transpose1d` / `conv_transpose2d` are both registered converters, so quantized
transposed convolution works end to end. Do not route around these.

**Never read table 1 without filtering by the installed PyTorch version.**

## 2. "Not quantizable" is not the same as "not convertible"

Of the 20 genuinely residual ops, **18 have an `aten::` converter registered**. They convert
successfully — as a float island:

```
... -> DEQUANTIZE -> <float op> -> QUANTIZE -> ...
```

The export succeeds and the numerics are fine. The cost is placement: on an Ethos-U target every
such island is a CPU operator, plus two conversion ops on each boundary.

Only two ops fail outright, because no converter is registered for them at all:

| Op | Missing converter |
|---|---|
| `atan` | `aten::atan` |
| `torch.nn.RNN` | `aten::rnn_tanh`, `aten::rnn_relu` |

These raise `Unsupported ops: ...` followed by `Cannot continue due to fatal error`
(`tinynn/converter/base.py:522-523`).

## 3. The 20 residual ops

| Op | Class | Outcome if left alone | Fix |
|---|---|---|---|
| `hardsigmoid` | A | float island | use `nn.Hardsigmoid` module |
| `layer_norm` | A | float island | use `nn.LayerNorm` module |
| `rsqrt` | B | float island | `.sqrt().reciprocal()` |
| `mm` | B | float island | `torch.matmul` |
| `norm` | B | float island | expand manually |
| `std` | B | float island | expand manually |
| `var` | B | float island | expand manually |
| `group_norm` | C | float island | write `QGroupNorm` |
| `torch.nn.GroupNorm` | C | float island | write `QGroupNorm` |
| `instance_norm` | C | float island | write `QInstanceNorm` |
| `torch.nn.InstanceNorm1d` | C | float island | write `QInstanceNorm` |
| `torch.nn.InstanceNorm2d` | C | float island | write `QInstanceNorm` |
| `sin` | D | float island | accept, or restructure |
| `cos` | D | float island | accept, or restructure |
| `exp` | D | float island | accept, or restructure |
| `log` | D | float island | accept, or restructure |
| `pow` | D | float island | accept, or restructure |
| `atan2` | D | float island | accept, or restructure |
| `atan` | E | **hard fail** | rewrite as `atan2(x, 1)` |
| `torch.nn.RNN` | E | **hard fail** | swap for `nn.LSTM` / `nn.GRU` |

---

## Class A — functional / module mismatch (free wins)

`Q_MODULES_MAPPING` maps unquantizable **modules** onto hand-written quantizable replacements:

```python
{nn.PReLU: QPReLU, nn.GLU: QGLU, nn.Hardsigmoid: QHardsigmoid,
 nn.SiLU: QSiLU, nn.LayerNorm: QLayerNorm, nn.RMSNorm: QRMSNorm}
```

A separate table, `FUNCTIONAL_MODULE_MAPPING`, rewrites *functional* calls into their module form
before quantization — but it only covers `relu`, `relu6`, `leaky_relu`, `elu`, `prelu`, `glu`,
`sigmoid`, `tanh`, `hardswish`, `silu`.

`hardsigmoid` and `layer_norm` are missing from it. So `nn.Hardsigmoid` quantizes and
`F.hardsigmoid` does not — the same computation, different outcome, purely because of which form
the model author wrote.

**Fix — change the model, not TinyNN:**

```python
# before: falls out of the quantized graph
x = F.hardsigmoid(x)
x = F.layer_norm(x, (C,))

# after: picks up QHardsigmoid / QLayerNorm automatically
self.act = nn.Hardsigmoid()
self.norm = nn.LayerNorm(C)
...
x = self.act(x)
x = self.norm(x)
```

Extending `FUNCTIONAL_MODULE_MAPPING` upstream also works in principle, but the rewrite site
(`quantizer.py:2196`) constructs the module with no arguments unless the op has a special case, so
`layer_norm` would need its `normalized_shape` threaded through first. The model-side change is
the safe route.

## Class B — decomposable into ops that already have a recipe

`REWRITE_QUANTIZABLE_RULE_LIST` is keyed on **op patterns**, and some patterns cover a composite
that the single fused op does not:

```python
{('truediv',), ('sum',), ('abs',), ('matmul',), ('clamp_min',),
 ('clamp_max',), ('bmm',), ('sqrt', 'reciprocal')}
```

Note the last entry: `sqrt` followed by `reciprocal` is handled, but `rsqrt` is not.

```python
# rsqrt -> sqrt + reciprocal
x = torch.rsqrt(x)          # float island
x = x.sqrt().reciprocal()   # quantized, with rewrite_quantizable=True

# mm -> matmul
x = torch.mm(a, b)          # float island
x = torch.matmul(a, b)      # quantized, with rewrite_quantizable=True
```

`norm` / `std` / `var` have no direct equivalent, but expand into `sub` / `mul` / `sum` / `sqrt`,
all of which are covered. `QLayerNorm.forward` is a worked example of exactly this expansion.

All of Class B needs `TFLiteConverter(..., rewrite_quantizable=True)`.

## Class C — needs a hand-written quantizable module

`GroupNorm` and `InstanceNorm` have no `Q*` counterpart, but they are the same shape of problem as
`LayerNorm`, which does. [`QLayerNorm`](../TinyNeuralNetwork/tinynn/graph/quantization/modules.py)
is roughly 50 lines and is the template: express the normalization as `FloatFunctional` ops so
each intermediate gets its own observer.

```python
# LayerNorm(x) = (x - mean(x)) * rsqrt(mean((x - mean(x))**2) + eps) * alpha + beta
mean = input.mean(self.mean_dims, keepdim=True)
diff = self.f_add_0.add(input, self.f_neg.mul_scalar(mean, -1.0).expand_as(input))
squarer_difference = self.f_mul_0.mul(diff, diff)
var = squarer_difference.mean(self.mean_dims, keepdim=True)
var_eps = self.f_add_1.add_scalar(var, self.eps)
...
```

`GroupNorm` differs only in `mean_dims` (reshape to `(N, G, -1)` and reduce the last two axes);
`InstanceNorm` reduces the spatial axes per channel. Register the result:

```python
from tinynn.graph.quantization.quantizer import Q_MODULES_MAPPING
Q_MODULES_MAPPING[nn.GroupNorm] = QGroupNorm
```

The upstream comment on `QLayerNorm` is worth heeding: splitting the op means every intermediate
carries its own quantization parameters, which costs accuracy. Calibrate and measure.

## Class D — transcendentals, float island is the realistic answer

`sin` `cos` `exp` `log` `pow` `atan2` all convert, all land as float islands, and none has a
TinyNN recipe. Options, in order of preference:

1. **Restructure so the op is not in the graph.** Positional encodings, `exp` in a decode step,
   `pow` in a normalization — these are frequently constant-foldable or movable to pre/post
   processing outside the exported graph. This is what the YOLOv9-T pipeline does with the DFL
   decode: cut the graph before it and run it on the host.
2. **Accept the CPU fallback** and check the cost in the vela report.
3. Approximate with a piecewise-linear / LUT construction built from quantizable ops. Expensive to
   build and to validate; only worth it if the op sits in a hot loop.

## Class E — hard failures

**`atan`.** No `aten::atan` converter exists, but `aten::atan2` is registered, and
`atan(x) == atan2(x, 1)`:

```python
x = torch.atan(x)                              # Unsupported ops: aten::atan
x = torch.atan2(x, torch.ones_like(x))         # converts (as a Class D float island)
```

This turns a fatal error into a float island — an improvement, not a full fix.

**`torch.nn.RNN`.** Neither `aten::rnn_tanh` nor `aten::rnn_relu` is registered, and unlike
`LSTM`/`GRU` there is no entry in `quantizable/`. `nn.RNN` also has `None` as its version value,
so no PyTorch upgrade will help. Replace it with `nn.LSTM` or `nn.GRU` (both quantize on torch
≥ 1.13 and both have converters), or unroll the recurrence into explicit per-step `matmul`s in the
model.

---

## Related knobs, for the cases this document does not cover

These handle *other* categories of quantization failure and are easy to confuse with the above:

| Knob | Where | What it does |
|---|---|---|
| `quantize_op_action={M: 'disable'}` | Quantizer config | leave op in float, insert Quant/DeQuant around it |
| `quantize_op_action={M: 'rewrite'}` | Quantizer config | leave op in float **but keep input/output qparams**, so the converter can re-quantize |
| layerwise config yml | `<work_dir>/<model>_q_config.yml` | per-op enable/disable, for mixed precision |
| `override_qconfig_func` | Quantizer config | per-layer observer / fake-quant choice |
| `set_quantizable_op_stats=True` | Quantizer config | inject known qparams for `softmax` / `log_softmax` (`KNOWN_QSTATS`) |
| `rewrite_quantizable=True` | TFLiteConverter | required by every Class B fix |
| `group_conv_rewrite=True` | TFLiteConverter | split grouped convs; required for Ethos-U |
| `unroll_rnn=True` | TFLiteConverter | expand LSTM/GRU into primitive ops |

## Maintenance

```
python tools/check_gaps.py
```

Re-derives the residual set against the installed PyTorch and the TinyNeuralNetwork checkout, and
reports drift from the 20 ops documented here. Rerun after a torch upgrade or a TinyNN update — a
newer PyTorch shrinks the set, a newer TinyNN may add entries to `Q_MODULES_MAPPING` or
`REWRITE_QUANTIZABLE_RULE_LIST`.
