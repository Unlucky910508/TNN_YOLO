# TinyNeuralNetwork knowledge base

Reference material for driving TinyNeuralNetwork conversions (PyTorch → INT8 TFLite), written for
a coding agent to grep.

| Document | Index key | Use it |
|---|---|---|
| [`quantization_gaps.md`](quantization_gaps.md) | operator name | **Before** converting — can this model reach a fully-quantized graph, and what has to change? |
| [`error_index.md`](error_index.md) | error message text | **During** — TinyNN raised something; what does it mean and what fixes it? |

Both are pinned to a specific TinyNeuralNetwork commit and PyTorch version; both carry a script
that re-derives them.

```
python tools/gen_error_index.py    # regenerates error_index.md + error_index.json
python tools/check_gaps.py         # verifies quantization_gaps.md against the installed env
```

Run both after updating the TinyNeuralNetwork checkout or bumping PyTorch.

## Why not just read the upstream docs

`TinyNeuralNetwork/docs/` has `quantization_support.md` and `op_matrix.md`, both generated. They
are the starting point, but they under-report in ways that matter:

- `quantization_support.md` table 1 mixes "never supported" with "supported since version X" in one
  list; on PyTorch 2.5.1, 8 of its rows are stale. `quantization_gaps.md` §1 covers this.
- `op_matrix.md` extracts operator limitations with a regex that only matches `assert` statements at
  one exact indentation level carrying a string message. It finds constraints for 29 of 223 ops;
  an AST walk finds them for 57, and picks up the constraints that live in the `tfl.*` emitters
  that conv/pool/RNN converters delegate to.
- Neither covers the tracer, the quantizer, or the graph optimization passes — roughly two thirds
  of the error sites on the conversion path.

## Not covered here

Failures where TinyNN exits 0 and the problem only shows up downstream (TFLite interpreter refuses
the model, vela places operators on the CPU, XNNPACK rejects a concat). Those are experience rather
than anything derivable from the source, and are worth accumulating in a third document as they are
encountered — see the two local patches in the TinyNeuralNetwork checkout for the first two cases.
