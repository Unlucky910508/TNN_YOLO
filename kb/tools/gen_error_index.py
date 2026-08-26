"""Extract every raise / assert / warn site in TinyNeuralNetwork into an error index.

The upstream `scripts/gen_op_docs.py` collects operator limitations with a regex that
only matches `assert` statements at exactly 8-space indentation that carry a string
message; it misses everything nested inside an `if`, everything without a message, and
every error raised outside an operator's `parse()` (tracer, quantizer, graph passes).

This walks the AST instead, so nesting and indentation do not matter, and it covers the
whole conversion stack rather than the operator converters alone.

Outputs (next to this script's parent directory):
  error_index.md    - human/agent readable, grouped by layer, indexed by message
  error_index.json  - same data, machine readable
"""

import argparse
import ast
import json
import os
import sys
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent
TINYNN_ROOT = KB_DIR.parent / "TinyNeuralNetwork"

# Layers of the conversion stack, matched against the path relative to `tinynn/`.
# Order matters: the first matching prefix wins.
LAYER_RULES = [
    ("graph/quantization/quantizable/", "quantizer"),
    ("graph/quantization/", "quantizer"),
    ("graph/", "tracer"),
    ("converter/operators/torch/", "converter-op"),
    ("converter/operators/tflite/", "tflite-emit"),
    ("converter/operators/", "converter-pass"),
    ("converter/", "converter-core"),
    ("util/", "util"),
]

# Not part of the pytorch -> tflite path.
SKIP_PREFIXES = (
    "prune/",
    "llm_quant/",
    "graph/configs/",  # offline codegen helpers, never run during a conversion
)

LAYER_ORDER = [
    "tracer",
    "quantizer",
    "converter-op",
    "tflite-emit",
    "converter-pass",
    "converter-core",
    "util",
]

LAYER_BLURB = {
    "tracer": (
        "Raised while tracing the model into TinyNN's rewritten graph, before any quantization."
        " Usually means the `forward` uses a construct the tracer cannot represent."
    ),
    "quantizer": (
        "Raised by `QATQuantizer` / `PostQuantizer` while rewriting the graph, fusing modules or"
        " preparing observers. The model is still PyTorch at this point."
    ),
    "converter-op": (
        "Raised while translating one TorchScript op into TFLite ops. These are the per-operator"
        " limitations: the op is registered, but this particular argument combination is not"
        " implemented."
    ),
    "tflite-emit": (
        "Raised while emitting a TFLite operator. Conv/pool/RNN-family converters delegate the actual"
        " constraint checking here, so an operator whose own `parse()` looks unconstrained can still"
        " fail in this layer."
    ),
    "converter-pass": (
        "Raised by a graph optimization pass (fusion, transpose elimination, group-conv rewrite,"
        " quantizable rewrite). Not attributable to a single source operator."
    ),
    "converter-core": "Raised by the converter driver itself (graph construction, tensor bookkeeping, export).",
    "util": "Raised by shared helpers.",
}


def layer_for(rel_path: str):
    for prefix in SKIP_PREFIXES:
        if rel_path.startswith(prefix):
            return None
    for prefix, layer in LAYER_RULES:
        if rel_path.startswith(prefix):
            return layer
    return None


def render_message(node):
    """Render a message expression back to text, keeping f-string placeholders as {expr}."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                try:
                    parts.append("{" + ast.unparse(value.value) + "}")
                except Exception:
                    parts.append("{...}")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        # "...%s..." % (x,)
        return render_message(node.left)
    return None


def render_condition(node):
    try:
        text = ast.unparse(node)
    except Exception:
        return None
    text = " ".join(text.split())
    return text if len(text) <= 200 else text[:197] + "..."


class Visitor(ast.NodeVisitor):
    """Collects error sites along with the class/function they live in."""

    def __init__(self, rel_path, layer):
        self.rel_path = rel_path
        self.layer = layer
        self.stack = []
        self.records = []
        # class name -> tfl.* emitter classes it constructs; conv/pool/rnn converters push
        # their real constraints down into these.
        self.emitters = {}

    # -- scope tracking ------------------------------------------------------
    def _scoped(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_ClassDef = _scoped
    visit_FunctionDef = _scoped
    visit_AsyncFunctionDef = _scoped

    def _add(self, node, kind, message, condition=None, fatal=True, authored=True):
        self.records.append(
            {
                "kind": kind,
                "message": message,
                "condition": condition,
                "fatal": fatal,
                "authored": authored,
                "file": self.rel_path,
                "line": node.lineno,
                "layer": self.layer,
                "scope": ".".join(self.stack),
                "cls": self.stack[0] if self.stack else None,
            }
        )

    # -- error sites ---------------------------------------------------------
    def visit_Assert(self, node):
        message = render_message(node.msg)
        condition = render_condition(node.test)
        authored = message is not None
        if message is None:
            # `assert False` with no message carries no information; anything else at least
            # tells the reader which invariant broke.
            if isinstance(node.test, ast.Constant) and node.test.value is False:
                return
            message = f"assert {condition}"
        self._add(node, "assert", message, condition, authored=authored)
        self.generic_visit(node)

    def visit_Raise(self, node):
        exc = node.exc
        if exc is None:
            return
        name, args = None, []
        if isinstance(exc, ast.Call):
            func = exc.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            args = exc.args
        else:
            name = getattr(exc, "id", None) or getattr(exc, "attr", None)
        message = render_message(args[0]) if args else None
        authored = message is not None
        if message is None:
            message = f"raise {name}"
        self._add(node, f"raise {name}", message, authored=authored)
        self.generic_visit(node)

    def visit_Call(self, node):
        func = node.func
        target = None
        if isinstance(func, ast.Attribute):
            owner = getattr(func.value, "id", None)
            if owner == "tfl" and func.attr.endswith("Operator") and self.stack:
                self.emitters.setdefault(self.stack[0], set()).add(func.attr)
            if func.attr == "warn" and owner in ("warnings", "warn"):
                target = "warning"
            elif func.attr == "error" and owner in ("log", "logger", "logging"):
                target = "log.error"
        if target and node.args:
            message = render_message(node.args[0])
            if message:
                self._add(node, target, message, fatal=(target == "log.error"))
        self.generic_visit(node)


def collect(tinynn_root: Path):
    records = []
    emitters = {}
    pkg = tinynn_root / "tinynn"
    for path in sorted(pkg.rglob("*.py")):
        rel = path.relative_to(pkg).as_posix()
        layer = layer_for(rel)
        if layer is None:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            print(f"  ! skipping {rel}: {exc}", file=sys.stderr)
            continue
        visitor = Visitor(f"tinynn/{rel}", layer)
        visitor.visit(tree)
        records.extend(visitor.records)
        for cls, names in visitor.emitters.items():
            emitters.setdefault(cls, set()).update(names)
    return records, emitters


def build_op_map(tinynn_root: Path):
    """converter class name -> [torchscript op keys], via the registration table."""
    sys.path.insert(0, str(tinynn_root))
    try:
        from tinynn.converter.operators.torch import OPERATOR_CONVERTER_DICT
    except Exception as exc:  # pragma: no cover - depends on the user's env
        print(f"  ! could not import OPERATOR_CONVERTER_DICT ({exc}); op column will be empty", file=sys.stderr)
        return {}, {}
    cls_to_ops = {}
    for op_key, cls in OPERATOR_CONVERTER_DICT.items():
        cls_to_ops.setdefault(cls.__name__, []).append(op_key)
    return cls_to_ops, OPERATOR_CONVERTER_DICT


def annotate(records, cls_to_ops):
    for rec in records:
        rec["ops"] = sorted(cls_to_ops.get(rec["cls"], [])) if rec["cls"] else []
    return records


def md_escape(text):
    return text.replace("|", "\\|").replace("\n", " ").strip()


def write_markdown(records, op_dict, cls_to_ops, emitters, out_path, tinynn_commit):
    by_layer = {}
    for rec in records:
        by_layer.setdefault(rec["layer"], []).append(rec)

    lines = []
    lines.append("<!-- Generated by kb/tools/gen_error_index.py. DO NOT EDIT BY HAND. -->\n")
    lines.append("# TinyNeuralNetwork Error Index\n\n")
    lines.append(
        "Every `assert`, `raise`, `warnings.warn` and `log.error` on the PyTorch -> TFLite path,"
        " indexed by the message text you actually see in a traceback.\n\n"
    )
    lines.append(f"- TinyNeuralNetwork commit: `{tinynn_commit}`\n")
    lines.append(f"- Error sites: **{len(records)}**\n")
    counts = ", ".join(f"{layer} {len(by_layer.get(layer, []))}" for layer in LAYER_ORDER if by_layer.get(layer))
    lines.append(f"- By layer: {counts}\n")
    lines.append(
        "\n> **How to use this.** Copy the exception message from the traceback and search this file"
        " for a distinctive fragment of it. Messages containing `{...}` are f-strings; the braces mark"
        " where runtime values are interpolated, so search the literal text around them.\n"
    )
    lines.append(
        "\n> A failure at an earlier layer means the later layers never ran. `tracer` errors happen"
        " before quantization, `quantizer` errors before conversion, and `converter-*` errors before"
        " the `.tflite` is written. Non-fatal rows (`warning`) do **not** stop the conversion - they"
        " are the ones that produce a model that exports cleanly but misbehaves downstream.\n\n"
    )

    lines.append("## Contents\n\n")
    for layer in LAYER_ORDER:
        if by_layer.get(layer):
            lines.append(f"- [{layer}](#{layer}) - {len(by_layer[layer])} sites\n")
    lines.append("- [Appendix: limitations by operator](#appendix-limitations-by-operator)\n\n")

    for layer in LAYER_ORDER:
        recs = by_layer.get(layer)
        if not recs:
            continue
        lines.append(f"## {layer}\n\n")
        lines.append(f"{LAYER_BLURB[layer]}\n\n")
        recs = sorted(recs, key=lambda r: (r["file"], r["line"]))
        authored = [r for r in recs if r["authored"]]
        bare = [r for r in recs if not r["authored"]]

        lines.append("| Message | Kind | Operator(s) | Source |\n")
        lines.append("|---|---|---|---|\n")
        for rec in authored:
            ops = ", ".join(f"`{o}`" for o in rec["ops"]) if rec["ops"] else ""
            fatal = "" if rec["fatal"] else " *(non-fatal)*"
            lines.append(
                f"| {md_escape(rec['message'])}{fatal} | `{rec['kind']}` | {ops} |"
                f" `{rec['file']}:{rec['line']}` |\n"
            )
        lines.append("\n")

        if bare:
            lines.append(f"<details>\n<summary>{len(bare)} bare assertions in this layer"
                         " (no message - the traceback shows only the failing expression)</summary>\n\n")
            lines.append("| Failing expression | Operator(s) | Source |\n")
            lines.append("|---|---|---|\n")
            for rec in bare:
                ops = ", ".join(f"`{o}`" for o in rec["ops"]) if rec["ops"] else ""
                lines.append(
                    f"| `{md_escape(rec['message'])}` | {ops} | `{rec['file']}:{rec['line']}` |\n"
                )
            lines.append("\n</details>\n\n")

    # Appendix: the completed op_matrix - every registered op, with every constraint.
    lines.append("## Appendix: limitations by operator\n\n")
    lines.append(
        "The same data re-indexed by TorchScript op. Two things upstream `docs/op_matrix.md` misses are"
        " included here: constraints nested inside conditionals, and constraints that live in the"
        " `tfl.*` emitter the converter delegates to (marked *via `tfl.X`*). An empty cell therefore"
        " really does mean no constraint was found.\n\n"
    )
    by_class = {}
    for rec in records:
        if rec["cls"]:
            by_class.setdefault(rec["cls"], []).append(rec)

    lines.append("| Operator | Converter class | Limitations |\n")
    lines.append("|---|---|---|\n")
    for op in sorted(op_dict):
        cls = op_dict[op].__name__
        seen, texts = set(), []
        for rec in sorted(by_class.get(cls, []), key=lambda r: r["line"]):
            if not rec["authored"]:
                continue
            if rec["message"] in seen:
                continue
            seen.add(rec["message"])
            texts.append(md_escape(rec["message"]))
        for emitter in sorted(emitters.get(cls, ())):
            for rec in sorted(by_class.get(emitter, []), key=lambda r: r["line"]):
                if not rec["authored"] or rec["message"] in seen:
                    continue
                seen.add(rec["message"])
                texts.append(f"{md_escape(rec['message'])} *(via `tfl.{emitter}`)*")
        lines.append(f"| `{op}` | `{cls}` | {'<br>'.join(texts)} |\n")
    lines.append("\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tinynn-root", default=str(TINYNN_ROOT))
    parser.add_argument("--out-dir", default=str(KB_DIR))
    args = parser.parse_args()

    tinynn_root = Path(args.tinynn_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    commit = os.popen(f"git -C {tinynn_root} rev-parse --short HEAD 2>/dev/null").read().strip() or "unknown"
    dirty = os.popen(f"git -C {tinynn_root} status --porcelain 2>/dev/null").read().strip()
    if dirty:
        commit += " (+local patches)"

    cls_to_ops, op_dict = build_op_map(tinynn_root)
    raw_records, emitters = collect(tinynn_root)
    records = annotate(raw_records, cls_to_ops)

    write_markdown(records, op_dict, cls_to_ops, emitters, out_dir / "error_index.md", commit)
    (out_dir / "error_index.json").write_text(
        json.dumps({"tinynn_commit": commit, "records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"{len(records)} error sites -> {out_dir / 'error_index.md'}")
    by_layer = {}
    for rec in records:
        by_layer[rec["layer"]] = by_layer.get(rec["layer"], 0) + 1
    for layer in LAYER_ORDER:
        if by_layer.get(layer):
            print(f"  {layer:16s} {by_layer[layer]}")


if __name__ == "__main__":
    main()
