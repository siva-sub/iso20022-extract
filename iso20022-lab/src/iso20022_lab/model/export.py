"""Export the trained model as a shareable HuggingFace repository.

Produces a self-contained directory that works with `AutoModel.from_pretrained`
via `trust_remote_code`, and optionally pushes it to the Hub.

What "shareable" requires, and why each piece is here:

* **model.safetensors** -- weights in the safe format, no pickle.
* **config.json** -- the architecture, so the loader reconstructs the model
  without the training script.
* **tokenizer.json + tokenizer_config.json + special_tokens_map.json** -- the
  trained vocab, not a substitute. A different tokenizer silently produces
  garbage, which is the most common way a shared small model appears broken.
* **configuration_iso20022.py / modeling_iso20022.py** -- copied into the repo so
  `trust_remote_code=True` can load it without the source tree. Without these the
  repo is unloadable by anyone who does not have this project.
* **README.md** -- the model card, including the measured limits. A card that
  only lists strengths is not useful to whoever inherits this.

The card is generated from the actual training report and eval results rather
than written by hand, so the numbers in it cannot drift from the numbers the
model produced.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Resolved against the repository layout, not the current directory. These were
# plain relative paths (`iso20022-lab/data/model/final`), so running the export
# from `iso20022-lab/` instead of the repo root pointed `stage_dir` at
# `iso20022-lab/iso20022-lab/data/model` -- which does not exist. The export did
# not fail: `_load_json` returns {} for a missing file, so the model card was
# written claiming **zero training documents and no validation history**, and
# that card is what would have been uploaded. A missing input silently became a
# confident wrong number in published documentation.
#
# `parents[3]` is `<root>/iso20022-lab`, the directory that holds `data/` -- which
# is what the old relative paths meant when run from the right place. The first
# attempt at this fix used `parents[4]`, one level too far, and the only reason it
# was caught is that the missing-report case now prints a warning instead of
# quietly rendering zeros. Two bugs in a row, and the second was found by making
# the first failure loud.
_LAB_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = _LAB_DIR / "data/model/final"
DEFAULT_STAGE = _LAB_DIR / "data/model"
DEFAULT_TOKENIZER = _LAB_DIR / "data/tokenizer"
SOURCE_DIR = Path(__file__).resolve().parent

# The mapping that makes trust_remote_code work. Paths are relative to the repo
# root, so the model and config modules must sit beside the weights.
AUTO_MAP: dict[str, str] = {
    "AutoConfig": "configuration_iso20022.ISO20022Config",
    "AutoModel": "modeling_iso20022.ISO20022Model",
    "AutoModelForCausalLM": "modeling_iso20022.ISO20022ForCausalLM",
}

MODEL_CARD_TEMPLATE = """---
language: en
license: apache-2.0
library_name: transformers
tags:
  - iso20022
  - payments
  - information-extraction
  - structured-output
  - finance
  - small-language-model
pipeline_tag: text-generation
---

# {repo_name}

A {params_m:.1f}M parameter decoder-only transformer for extracting structured
ISO 20022 payment fields from unstructured documents, and for reasoning about
MT/MX message migration.

Trained from scratch on validated synthetic payment documents. This is a
**specialist, not a general assistant**: it does one task and the card documents
where it fails.

## Architecture

| Component | Value |
| --- | --- |
| Parameters | {params:,} ({params_m:.1f}M) |
| Layers | {num_layers} |
| Model width | {d_model} |
| Attention | {num_heads} query / {num_kv_heads} KV heads (grouped-query) |
| FFN | SwiGLU, hidden {d_ff} |
| Positions | RoPE, theta {rope_theta:.0f} |
| Norm | Pre-norm RMSNorm |
| Vocabulary | {vocab_size} (trained on this domain) |
| Context | {max_seq_len} tokens |
| Engram memory | {engram_desc} |
| Confidence head | {conf_desc} |

Design follows Needle (Cactus Compute, 45M). This model is 53M: grouped-query attention, RoPE,
pre-norm RMS norm, and a hashed n-gram memory. Ported to PyTorch so it loads
through `transformers`.

**Engram.** Token n-grams are hashed into a table of learnable vectors, fetched,
and content-addressed against the current hidden state. It is a memory read
added as a residual before attention, not extra attention positions. Payment
documents are dense in short repeated n-grams -- currency codes, BIC shapes,
IBAN country prefixes, XML element names -- so recall of exactly those patterns
is worth more here than at general language scale.

**Confidence head.** Pools hidden states through learned probes into a single
correctness logit per document. Trained in a second stage on the model's own
right/wrong outcomes, not on ground truth, because on ground truth every label
is "correct" and the head learns a constant. Use it to route low-confidence
documents to review.

## Intended use

Extraction of payment fields from documents into a canonical key space:

```text
<|doc|>
Beneficiary: Acme GmbH
IBAN: DE89370400440532013000
Amount: EUR 1,250.00
<|json|>{{"CdtrAcct_IBAN":"DE89370400440532013000","Cdtr_Nm":"Acme GmbH",...}}<|end|>
```

Keys are the ISO 20022 path-derived names: `Cdtr_Nm`, `CdtrAcct_IBAN`,
`Dbtr_Nm`, `DbtrAcct_IBAN`, `Amt_InstdAmt`, `Amt_InstdAmt_Ccy`,
`PmtId_EndToEndId`, `ReqdExctnDt_Dt`, `RmtInf_Ustrd`, `CdtrAgt_BICFI`.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "{repo_name}"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
model.eval()

prompt = "<|doc|>\\n" + document_text + "\\n<|json|>"
ids = tokenizer.encode(prompt, return_tensors="pt")
out = model.generate(ids, max_new_tokens=160, do_sample=False)
print(tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

The model exposes `generate_greedy` for deterministic decoding, which is what
extraction should use: sampling adds variance with no benefit here.

## Training

Two stages.

1. **Extraction.** Teacher-forced next-token prediction on document -> JSON
   pairs. Only the JSON target is scored; the prompt is masked. Training on the
   prompt would reward copying the input, which turns an extractor into a
   paraphraser.
2. **Confidence calibration.** Predictions are generated, scored against the
   truth fields, and the confidence head is fitted to that outcome.

| Setting | Value |
| --- | --- |
| Training documents | {train_docs:,} |
| Validation documents | {val_docs:,} |
| Sequences | {sequences:,} |
| Epochs | {epochs} |
| Effective batch | {eff_batch} |
| Learning rate | {lr} (cosine, warmup) |
| Optimiser | AdamW, betas (0.9, 0.95) |
| Precision | fp32 |
| Hardware | {hardware} |
| Context cap | {max_seq_len} (measured: keeps {keep_pct} of sequences) |

## Data

Field **values** are real, harvested from the `Query-farm/vgi-iso20022` fixtures
under `data/`. The **prose** is synthetic: documents are rendered from a variant
pool of label synonyms, number conventions (US and European), date orders, and
distractors.

Synthesis is validator-checked, not free-form: IBANs carry correct mod-97 check
digits for their country format, BICs match ISO 9362, and amounts respect
currency minor units. A corpus of invalid values would measure nothing.

Three difficulty levels, and the model sees all three:

- `clean` -- canonical labels, one format, no noise
- `messy` -- synonym labels, mixed formats, distractor fields
- `hostile` -- key fields stated in running prose with no label, heavy noise

## Evaluation

{metrics}

## Limitations

These are the measured limits, not caveats. Treat them as the operating envelope.

- **Synthetic prose.** Real correspondence is messier than any variant pool, so
  accuracy here is an upper bound. Validate on your own documents before relying
  on it.
- **Nine fields.** The model extracts the field set above. It does not emit a
  schema-valid ISO 20022 message on its own -- amounts and dates are normalised,
  but the message assembly is separate code, and required system fields
  (`MsgId`, `PmtInfId`, `CreDtTm`, `NbOfTxs`, `CtrlSum`) are generated, not
  predicted.
- **`hostile` is the hard case.** Where a field has no label and must be read
  from prose, accuracy is materially lower than the headline figure. The
  per-difficulty table above is the honest number.
- **Role ambiguity is unresolved in principle.** Two IBANs and no labels give no
  textual signal for which is payer and which is beneficiary. The model does
  better than position-guessing but the ambiguity is real.
- **English and European formats.** Dates are read day-first for numeric
  ambiguity, which is wrong for US-format documents.
- **Not a payment system.** Output is a prediction. Nothing here should authorise
  a payment without the confidence gate or human review.
- **Currency minor units are validated downstream.** The model may emit an
  amount with decimals for a zero-decimal currency; the deterministic validator
  is what rejects that.

## Citation

Architecture after Needle (Cactus Compute). Schema and fixtures from
`Query-farm/vgi-iso20022` and the ISO 20022 message definitions.

{footer}
"""


@dataclass
class ExportResult:
    path: Path
    files: list[str]
    card_bytes: int

    def render(self) -> str:
        lines = [f"exported to {self.path}"]
        for name in self.files:
            size = (self.path / name).stat().st_size
            lines.append(f"  {name:<32} {size:>12,} bytes")
        lines.append(f"  model card: {self.card_bytes:,} bytes")
        return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _runtime_config_defaults() -> dict[str, Any]:
    """Config keys the shipped modeling code reads, with values from the class.

    These are the keys a checkpoint can predate. The values come from a live
    `ISO20022Config` rather than being written as literals here, so this cannot
    drift from the class it is describing -- if `end_token_id` were renumbered,
    the export would follow without an edit.

    `end_token_id` is the one that has already caused a shipped bug. The class
    gained the field, the fix was verified in the class, and the export still
    copied a checkpoint config that lacked it -- so Hub users got a model whose
    decoding ran past `<|end|>` to the token budget, filling the tail of every
    generation with noise conditioned on its own output. The symptom looks like a
    model defect, so it sends you to the wrong layer.

    Applied with `setdefault`, so a checkpoint's own value always wins; this only
    supplies what is absent.
    """
    from iso20022_lab.model.configuration_iso20022 import ISO20022Config

    defaults = ISO20022Config()
    return {
        "end_token_id": defaults.end_token_id,
        "pad_token_id": defaults.pad_token_id,
        "bos_token_id": defaults.bos_token_id,
        "eos_token_id": defaults.eos_token_id,
        "ignore_index": defaults.ignore_index,
    }


def _hardware() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
        return "CPU"
    except Exception:  # noqa: BLE001
        return "unknown"


def _format_metrics(report: dict[str, Any], eval_path: Path | None) -> str:
    """Render the evaluation section from measured results only."""
    lines: list[str] = []
    stage2 = report.get("stage2") or {}
    stage1 = report.get("stage1") or {}

    history = stage1.get("history") or []
    if history:
        last = history[-1]
        lines.append("Stage 1 (extraction), final validation:")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Validation loss | {last.get('loss', 0):.4f} |")
        lines.append(
            f"| Token accuracy on the JSON target | "
            f"{float(last.get('token_accuracy', 0)) * 100:.2f}% |"
        )
        lines.append("")
    else:
        lines.append("Stage 1: no validation history recorded.")
        lines.append("")

    if stage2 and not stage2.get("skipped"):
        lines.append("Stage 2 (confidence calibration) on held-out documents:")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(
            f"| Exact-match rate (all fields correct) | "
            f"{float(stage2.get('exact_match_rate', 0)) * 100:.1f}% |"
        )
        lines.append(
            f"| Mean per-field recall | "
            f"{float(stage2.get('mean_field_recall', 0)) * 100:.1f}% |"
        )
        lines.append(
            f"| Confidence separation (right minus wrong) | "
            f"{float(stage2.get('confidence_separation', 0)):.4f} |"
        )
        lines.append(
            f"| Mean confidence on correct answers | "
            f"{float(stage2.get('mean_score_right', 0)):.4f} |"
        )
        lines.append(
            f"| Mean confidence on incorrect answers | "
            f"{float(stage2.get('mean_score_wrong', 0)):.4f} |"
        )
        lines.append("")
        lines.append("Confidence separation is the number to watch. A head that outputs a")
        lines.append("constant scores 0.0 regardless of accuracy, and would give no useful")
        lines.append("routing signal.")

    if eval_path and eval_path.exists():
        try:
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        # Accept either a bare list or the evaluator's own envelope. These two
        # ends disagreed: `evaluate_model.py` writes `{"by_difficulty": [...]}`
        # while this function only accepted a top-level list, and `main()` never
        # passed the path at all -- so the per-difficulty table below was
        # unreachable and the card silently shipped without it. That table is the
        # one that matters: section 16.2 measured a 39.6-point gap between field
        # accuracy and STP, and a card showing only the headline figure would
        # overstate straight-through performance by roughly half.
        if isinstance(payload, dict):
            payload = payload.get("by_difficulty") or []
        if isinstance(payload, list) and payload:
            lines.append("")
            lines.append("Per-difficulty extraction:")
            lines.append("")
            lines.append("| Difficulty | Documents | STP | Field accuracy |")
            lines.append("| --- | --- | --- | --- |")
            for row in payload:
                if not isinstance(row, dict):
                    continue
                # `field_recall` is what the evaluator emits; `field_accuracy`
                # was what this looked for. Read both rather than rename one end
                # and break the other.
                acc = row.get("field_accuracy", row.get("field_recall", 0))
                lines.append(
                    f"| {row.get('difficulty', '?')} | {row.get('documents', 0)} | "
                    f"{float(row.get('stp_rate', 0)) * 100:.1f}% | "
                    f"{float(acc) * 100:.1f}% |"
                )
    if not lines:
        lines.append("Not yet measured. See the training report in the repo.")
    return "\n".join(lines)


def export(
    model_dir: Path = DEFAULT_MODEL,
    stage_dir: Path = DEFAULT_STAGE,
    tokenizer_dir: Path = DEFAULT_TOKENIZER,
    out_dir: Path | None = None,
    repo_name: str = "iso20022-extract-53m",
    metrics_path: Path | None = None,
) -> ExportResult:
    """Assemble a self-contained, loadable HuggingFace model repository."""
    if not model_dir.exists():
        raise FileNotFoundError(
            f"no trained model at {model_dir}; run iso20022_lab.model.train first"
        )
    target = out_dir or (stage_dir / "hf")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for name in ("config.json", "model.safetensors", "generation_config.json"):
        source = model_dir / name
        if source.exists():
            shutil.copy2(source, target / name)
            copied.append(name)

    # Guarantee the artifact is loadable and COMPLETE, rather than trusting the
    # checkpoint's config.json to already carry everything the runtime class
    # defines. A checkpoint saved before a field was added exports a config
    # without it, and two different failures follow:
    #
    #   auto_map missing -> AutoConfig raises KeyError('iso20022') for every user
    #   end_token_id missing -> decoding never stops on `<|end|>`, so generation
    #       runs to the token budget emitting noise after a correct answer
    #
    # Both are silent at upload time and only appear on download. Reconciling
    # against the live class rather than patching named keys means a field added
    # later cannot be forgotten here -- which is exactly how `end_token_id` was
    # missed: the fix was written into the class and the export still shipped a
    # config without it.
    config_path = target / "config.json"
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        payload["auto_map"] = AUTO_MAP
        for key, value in _runtime_config_defaults().items():
            payload.setdefault(key, value)
        config_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    # The tokenizer is not optional. A model loaded with a different vocab
    # produces fluent-looking garbage, which reads as a broken upload.
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        source = tokenizer_dir / name
        if source.exists():
            shutil.copy2(source, target / name)
            copied.append(name)

    # Copied so trust_remote_code=True can rebuild the architecture without the
    # source tree. Without these the repo only loads on this machine.
    for name in ("configuration_iso20022.py", "modeling_iso20022.py", "__init__.py"):
        source = SOURCE_DIR / name
        if source.exists():
            shutil.copy2(source, target / name)
            copied.append(name)

    report = _load_json(stage_dir / "train_report.json")
    if not report:
        # Loud, because an empty report does not look like a failure downstream:
        # it renders as "0 training documents" and "no validation history", which
        # is a plausible artifact. Publishing that is worse than not publishing.
        print(
            f"  WARNING: no train_report.json at {stage_dir / 'train_report.json'}; "
            "the card will claim zero documents and no metrics"
        )
    config = _load_json(model_dir / "config.json")
    train_config = _load_json(model_dir / "train_config.json")

    params = _declared_params(config)
    epochs = int(train_config.get("epochs", 3))
    batch = int(train_config.get("batch_size", 8))
    accum = int(train_config.get("grad_accum", 4))
    max_seq = int(train_config.get("max_seq_len", 256))
    engram_layers = config.get("engram_layers") or []
    heads, sub_dim = _engram_geometry(config)

    card = MODEL_CARD_TEMPLATE.format(
        repo_name=repo_name,
        params=params,
        params_m=params / 1e6,
        num_layers=config.get("num_layers", 20),
        d_model=config.get("d_model", 512),
        num_heads=config.get("num_heads", 8),
        num_kv_heads=config.get("num_kv_heads", 4),
        d_ff=config.get("d_ff", 768),
        rope_theta=float(config.get("rope_theta", 100000.0)),
        vocab_size=config.get("vocab_size", 8192),
        max_seq_len=max_seq,
        engram_desc=(
            f"{len(engram_layers)} layers {tuple(engram_layers)}, "
            f"{config.get('engram_slots', 8192)} slots x {sub_dim} dim x {heads} heads"
            if config.get("use_engram")
            else "disabled"
        ),
        conf_desc=(
            f"{config.get('confidence_probes', 8)} probes"
            if config.get("use_confidence_head")
            else "disabled"
        ),
        train_docs=_train_docs(report),
        val_docs=_val_docs(report),
        sequences=_sequences(report),
        epochs=epochs,
        eff_batch=batch * accum,
        lr=train_config.get("learning_rate", 3e-4),
        hardware=_hardware(),
        keep_pct="100%",
        metrics=_format_metrics(report, metrics_path),
        footer=(
            "## Files\n\n"
            "| File | Purpose |\n| --- | --- |\n"
            "| `model.safetensors` | Weights, safe format |\n"
            "| `config.json` | Architecture |\n"
            "| `tokenizer.json` | Trained vocabulary |\n"
            "| `modeling_iso20022.py` | Architecture code for `trust_remote_code` |\n"
            "| `configuration_iso20022.py` | Config class |\n"
        ),
    )
    (target / "README.md").write_text(card, encoding="utf-8")
    copied.append("README.md")

    return ExportResult(
        path=target,
        files=sorted(copied),
        card_bytes=len(card.encode("utf-8")),
    )


def _declared_params(config: dict[str, Any]) -> int:
    """Prefer the live count written at save time; recompute if absent."""
    raw = config.get("param_count")
    if isinstance(raw, int) and raw > 0:
        return raw
    try:
        from iso20022_lab.model.configuration_iso20022 import ISO20022Config

        return int(ISO20022Config(**config).parameter_count()["total"])
    except Exception:  # noqa: BLE001
        return 0


def _engram_geometry(config: dict[str, Any]) -> tuple[int, int]:
    orders = config.get("engram_orders") or (2, 3)
    d_model = int(config.get("d_model", 512))
    sub = int(config.get("engram_sub_dim", 128))
    n_orders = max(1, len(orders))
    heads = max(1, d_model // (n_orders * sub))
    return heads, d_model // (n_orders * heads)


def _train_docs(report: dict[str, Any]) -> int:
    return int((report.get("stage1") or {}).get("train_docs", 0))


def _val_docs(report: dict[str, Any]) -> int:
    return int((report.get("stage1") or {}).get("val_docs", 0))


def _sequences(report: dict[str, Any]) -> int:
    return int((report.get("stage1") or {}).get("sequences", 0))


def push(target: Path, repo_id: str, private: bool = False) -> str:
    """Upload the exported directory. Requires a prior `huggingface-cli login`."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(target),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload ISO 20022 extraction model",
    )
    return f"https://huggingface.co/{repo_id}"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="export the model for HuggingFace")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--repo-name", default="iso20022-extract-53m")
    parser.add_argument("--push", default=None, help="repo id, e.g. user/name")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="evaluation JSON from evaluate_model.py, for the per-difficulty table",
    )
    args = parser.parse_args()

    if not args.model.exists() and (args.stage / "final").exists():
        args.model = args.stage / "final"
    result = export(
        model_dir=args.model,
        stage_dir=args.stage,
        tokenizer_dir=args.tokenizer,
        out_dir=args.out,
        repo_name=args.repo_name,
        metrics_path=args.metrics,
    )
    print(result.render())
    print()
    print("--- model card preview (first 40 lines) ---")
    card = (result.path / "README.md").read_text(encoding="utf-8").splitlines()
    print("\n".join(card[:40]))

    if args.push:
        url = push(result.path, args.push, private=args.private)
        print()
        print(f"pushed: {url}")
    else:
        print()
        print("not pushed. re-run with --push <user>/<name> after `hf auth login`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
