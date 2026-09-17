"""Training for the ISO 20022 extraction model.

Two stages, because they answer different questions and cannot be merged:

**Stage 1 -- extraction.** Teacher-forced next-token prediction on
document -> JSON pairs. Only the JSON target is scored; the prompt is masked.
Training on the prompt would reward the model for copying the input, which is
exactly the failure mode that turns an extraction model into a paraphraser.

**Stage 2 -- confidence calibration.** The confidence head cannot be trained in
stage 1. On ground-truth targets every answer is correct, so the label is 1.0 for
every example, the head learns to output a constant, and it carries no
information at all. To learn "am I right here" the head needs examples where the
model is *wrong*, which only exist once the model is generating. So stage 2
samples predictions, compares each against the truth fields, and fits the head to
the outcome.

That second stage is the whole point of the head: PROCESS.md 2.4 removed the
confidence gate and 16.3 makes hallucination the decisive risk, so a calibrated
"this answer is unreliable, route it to a human" signal is what makes the
pipeline safe to automate. A constant 1.0 score would leave the project where it
started.
"""

# pyright: reportPrivateImportUsage=false
#
# This directive must be a real comment before any code, not text inside the
# docstring above -- pyright reads comments, so a pragma quoted in the module
# docstring has no effect at all.
#
# Why it is needed: torch/__init__.py does `from torch._C import *`, where
# torch._C is a compiled extension with no stub. pyright cannot follow that
# wildcard, so it reports every `torch.tensor` / `torch.randperm` / `torch.long`
# as a private import from `torch._C._VariableFunctions`. All of them are present
# in torch.__all__ (910 entries) and available at runtime. Importing from
# `torch._C._VariableFunctions` to satisfy the checker would be private API that
# breaks on torch upgrades, so the public API stays and this one rule is silenced
# for this file only.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# From `fields`, not `acceptance`. Importing these from the gate closed a cycle --
# acceptance -> workflow -> model.evaluate_model -> model.train -> acceptance --
# which pyright reported as `Cycle detected in import chain`. A deferred import
# would have hidden the report while leaving the layering wrong: the real problem
# was that pure string parsing lived in the module that knows about trained
# models. `fields` imports nothing from this package, so the cycle cannot recur.
from iso20022_lab.fields import (
    CANONICAL_KEYS,
    count_duplicate_keys,
    parse_strict,
    stratified_sample,
)
from iso20022_lab.model.configuration_iso20022 import ISO20022Config
from iso20022_lab.model.modeling_iso20022 import ISO20022ForCausalLM, parameter_count
from iso20022_lab.paths import data_path
from iso20022_lab.model.tokenizer import (
    EOS_ID,
    PAD_ID,
    format_example,
    prompt_for,
)

DEFAULT_OUT = data_path("model")


class Example(TypedDict):
    """One training document.

    Typed rather than `dict[str, str]`: the payload genuinely nests a mapping
    under `values`, and flattening the annotation to `dict[str, str]` let a
    string reach `_encode` where a mapping was expected.
    """

    text: str
    values: dict[str, str]
    difficulty: str


@dataclass
class TrainConfig:
    """Everything that shapes a run. Serialised next to the checkpoint."""

    # Sized from the measured length distribution, not guessed. Across 1200
    # documents: p50 193, p99 231, max 239 tokens. A 192 cap silently discarded
    # 50% of the training data; 256 keeps 100% with headroom and is a third smaller
    # than 384, which buys back the activation memory that pays for batch 8.
    max_seq_len: int = 256
    # Measured peaks at seq 256: batch 4 -> 1.57 GB, batch 8 -> 2.46 GB, both inside
    # the 4 GB card. Batch 16 does not fit. Effective batch is 8 * 4 = 32.
    batch_size: int = 8
    grad_accum: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_frac: float = 0.03
    min_lr_frac: float = 0.1
    epochs: int = 3
    max_grad_norm: float = 1.0
    eval_every: int = 200
    save_every: int = 400
    seed: int = 0
    confidence_weight: float = 0.1
    # Off by default: measured batch 8 at seq 256 peaks at 2.46 GB of a 4 GB card,
    # so recomputation would cost ~1.5x speed for headroom the run does not need.
    # Left as a switch for larger batches or a longer context.
    gradient_checkpointing: bool = False
    device: str = "cuda"
    log_every: int = 20
    # Generation-based evaluation. Teacher-forced token accuracy is cheap but does
    # not predict whether the artifact works. At every step the model is handed the
    # *correct* prefix, so it scores well while free-running generation collapses;
    # the 53M run reached 82.14% token accuracy and extracted 1 field in 120. That
    # metric read like progress on a model that did not function.
    #
    # These settings buy a real measurement instead: `eval_generate` documents are
    # decoded greedily and scored against truth at every `eval_every` step. It is a
    # sample rather than the full validation set purely for cost -- decode is ~13s
    # per document here, so 8 documents adds ~2 minutes per evaluation. Set to 0 to
    # skip, which is the right choice for a smoke test.
    eval_generate: int = 8
    eval_max_new_tokens: int = 160

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class ExtractionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Document -> JSON sequences, tokenised once up front.

    The prompt is masked with -100 in `labels` so `cross_entropy` ignores it.
    Sequences are left unpadded here; `collate` pads per batch instead.
    """

    def __init__(
        self,
        examples: list[Example],
        tokenizer: Any,
        max_seq_len: int,
    ):
        # Assigned before the loop: `_encode` reads it, and without this the
        # first call raised AttributeError on an attribute that never existed.
        self.max_seq_len = max_seq_len
        self.samples: list[tuple[list[int], list[int]]] = []
        self.skipped = 0
        for item in examples:
            text = item.get("text", "")
            values = item.get("values", {})
            if not text or not values:
                self.skipped += 1
                continue
            encoded = self._encode(text, values, tokenizer)
            if encoded is None:
                self.skipped += 1
                continue
            self.samples.append(encoded)
            if len(self.samples) and len(self.samples) % 2000 == 0:
                print(f"    tokenised {len(self.samples)}", flush=True)

    def _encode(
        self, text: str, values: dict[str, str], tokenizer: Any
    ) -> tuple[list[int], list[int]] | None:
        sequence = format_example(text, values)
        ids = tokenizer.encode(sequence, add_special_tokens=False)
        if len(ids) > self.max_seq_len:
            return None
        prompt_ids = tokenizer.encode(prompt_for(text), add_special_tokens=False)
        # -100 is the ignore index for cross_entropy, so the prompt contributes
        # nothing to the loss.
        labels = [-100] * min(len(prompt_ids), len(ids))
        labels += ids[len(labels) :]
        if all(label == -100 for label in labels):
            return None  # nothing left to learn from
        # Unpadded: collate pads each batch to its own longest sequence.
        return (list(ids), labels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids, labels = self.samples[index]
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad each batch to its own longest sequence, not to `max_seq_len`.

    Padding everything to the global maximum is what made this fail on a 4 GB
    card: documents here run about 150 tokens, so a fixed 384-wide batch spent
    most of its activation memory on pad tokens and the run died with
    CUBLAS_STATUS_EXECUTION_FAILED once the allocator ran out. Padding to the
    batch maximum removes that waste and changes nothing the model sees.

    Width is rounded up to a multiple of 8 to keep shapes kernel-friendly, and
    pad positions carry -100 so they never contribute loss.
    """
    limit = max(ids.numel() for ids, _ in batch)
    width = ((limit + 7) // 8) * 8
    out_ids: list[torch.Tensor] = []
    out_labels: list[torch.Tensor] = []
    for ids, labels in batch:
        pad = width - ids.numel()
        out_ids.append(F.pad(ids, (0, pad), value=PAD_ID))
        out_labels.append(F.pad(labels, (0, pad), value=-100))
    return torch.stack(out_ids), torch.stack(out_labels)


def cosine_schedule(
    step: int, total_steps: int, warmup_steps: int, base_lr: float, min_frac: float
) -> float:
    """Linear warmup then cosine decay to `min_frac * base_lr`."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return base_lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def build_examples(count: int, seed: int, levels: int = 3) -> list[Example]:
    """Synthesise documents with exact ground truth.

    The corpus `truth` is canonical, so it is directly trainable -- no teacher
    labelling step is needed to get targets. Volume comes from validated value
    synthesis, and the structural variety comes from the renderer's difficulty
    levels, so the model sees clean, messy and hostile layouts rather than one.

    `count` is the number of distinct **value-sets**, and `levels` is how many
    difficulty levels each one is rendered at. Those two together decide the
    document count, and the ratio between them is the generalisation lever.

    `levels=3` (the default) renders every value-set clean, messy and hostile:
    4,000 value-sets become 12,000 documents, and each value-set is seen three
    times. The repetition reinforces the document->JSON mapping for that exact
    value-set while contributing no new values, and generalisation is bounded by
    distinct value-sets rather than by documents. Measured on a model trained
    that way: 97.5% field recall against a held-out slice of its own value draw,
    61.2% against a fresh one.

    `levels=1` gives each value-set a single difficulty, cycling through them, so
    the same document budget carries three times as many distinct value-sets --
    no value-set seen twice. Difficulty coverage is unchanged, because every
    third value-set still gets clean, messy and hostile.

    Splitting the budget across more *seeds* was tried as a fix and measured as a
    no-op: `generate_many` draws without replacement, so `count` value-sets is
    `count` value-sets however they are partitioned.
    """
    from iso20022_lab.corpus import build_corpus
    from iso20022_lab.synth import generate_many

    values = generate_many(count, seed=seed)
    corpus = build_corpus(values, per_seed=1, seed=seed, one_level_per_seed=levels <= 1)
    return [
        {"text": doc.text, "values": doc.truth, "difficulty": doc.difficulty}
        for doc in corpus.documents
    ]


def _value_key(item: Example) -> str:
    """Stable identity of an example's *answer*, independent of its wording."""
    values = item.get("values") or {}
    return "|".join(f"{k}={values[k]}" for k in sorted(values))


def split_examples(
    examples: list[Example], val_frac: float = 0.05, seed: int = 0
) -> tuple[list[Example], list[Example]]:
    """Split by value-set, on a hash that does not depend on corpus size.

    Two properties matter here and the previous version had neither.

    It assigned per *document* using ``torch.randperm``. ``build_examples`` makes
    three documents from every value-set -- clean, messy and hostile -- so
    holding out the clean variant still left that same answer in the training
    data through its two siblings. Measured on the 12,000-document corpus, **600
    of 600** validation answers were present in training, so the validation set
    could not distinguish extraction from recall of an answer already seen.

    And ``randperm(len(items))`` depends on how many items there are, so the
    split depended on corpus size: ``build_examples(4000)`` and
    ``build_examples(200)`` produced validation sets overlapping in **2
    documents**. Two runs at different ``--docs`` compared two different
    validation sets while both reporting "the same seed".

    Assigning whole value-sets by a hash of their own contents fixes both. The
    assignment is a function of the value-set and the seed alone, so it is stable
    across corpus sizes and arrival orders, and every variant of a value-set
    lands on the same side. Difficulty coverage is preserved for free, because
    each held-out value-set contributes all of its variants.

    **This still holds out a slice of one value *draw*.** ``generate_many`` draws
    N value-sets from a large random space, and ``generate_many(60, seed=0)`` is a
    subset of ``generate_many(4000, seed=0)`` -- so the held-out value-sets are
    neighbours of the training ones, drawn from the same pool in the same order.
    A model that memorises value instances scores well on them anyway. For the
    honest generalisation number, validate against a **fresh draw** with
    ``holdout_examples`` instead; see the note there.
    """
    groups: dict[str, list[Example]] = {}
    for item in examples:
        groups.setdefault(_value_key(item), []).append(item)

    train: list[Example] = []
    val: list[Example] = []
    for key in sorted(groups):
        digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
        bucket = int(digest[:8], 16) % 1000
        (val if bucket < val_frac * 1000 else train).extend(groups[key])
    return train, val


def holdout_examples(count: int, seed: int) -> list[Example]:
    """Build a validation corpus from a value draw disjoint from training's.

    Held-out slices of the training draw cannot measure generalisation here, and
    the size of the error was measured rather than argued. Training on
    ``seed=0`` and scoring against a held-out slice of that same draw gives
    **97.5%** field recall. Scoring the same model against a fresh draw -- same
    generator, same difficulty mix, ``seed=1`` -- gives **61.2%**. The gap is
    entirely the values: holding the values fixed and re-rendering the prose with
    the other seed leaves recall at 97.5%, so the prose distribution is not the
    variable. The model had memorised value instances.

    That is feasible rather than surprising: ``generate_many(4000, seed=0)``
    produces 4,000 value-sets of IBANs, amounts and remittance strings, the split
    holds out 216 of them, and 53M parameters can store all of it. A held-out
    slice of one draw stays inside the region the model has already seen.

    This is why the number matters. Every report this project produced quoted the
    within-draw figure, and every one of them overstated the model.

    ``levels=1`` rather than the default three, because the metric samples a
    handful of documents per evaluation and building three times as many as it
    will ever look at costs generation time for nothing. Difficulty coverage is
    still complete: a single level per value-set cycles through all three.
    """
    return build_examples(count, seed=seed, levels=1)


@torch.no_grad()
def evaluate(
    model: ISO20022ForCausalLM,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: str,
    max_batches: int = 40,
) -> dict[str, float]:
    """Teacher-forced validation loss and next-token accuracy.

    This was documented as returning "exact-match rate", which it never computed.
    That stale line is worth remembering: the number this function does return sat
    in the training log for two hours looking like evidence of progress on a model
    that could not produce a valid document.

    Kept for divergence detection, not for judging quality. Use
    `generation_metrics` for that.
    """
    model.eval()
    total_loss = 0.0
    batches = 0
    correct = 0
    total = 0
    for ids, labels in loader:
        ids = ids.to(device)
        labels = labels.to(device)
        out = model(input_ids=ids, labels=labels)
        loss = out["loss"]
        if loss is not None:
            total_loss += float(loss)
            batches += 1
        logits = out["logits"]
        assert isinstance(logits, torch.Tensor)
        predictions = logits[:, :-1, :].argmax(dim=-1)
        targets = labels[:, 1:]
        mask = targets.ne(-100)
        if mask.any():
            correct += int((predictions[mask] == targets[mask]).sum())
            total += int(mask.sum())
        if batches >= max_batches:
            break
    model.train()
    return {
        "loss": total_loss / max(1, batches),
        "token_accuracy": correct / total if total else 0.0,
    }


@torch.no_grad()
def model_device(model: Any) -> str:
    """Where a model's parameters actually live.

    Callers were trusted to pass the device the model was on, and one of them was
    wrong: the checkpoint verifier passed ``"cpu"`` for the live model while a GPU
    training run held it on ``cuda:0``. The mismatch does not raise at the call,
    it raises deep inside ``nn.Embedding`` as
    "Expected all tensors to be on the same device, but got index is on cpu" --
    which killed a four-hour run at its first checkpoint, after it had already
    trained past the point of being cheap to redo.

    Deriving it means the argument cannot be wrong, because there is no argument.
    """
    try:
        return str(next(model.parameters()).device)
    except (StopIteration, AttributeError, TypeError):
        return "cpu"


def generation_metrics(
    model: ISO20022ForCausalLM,
    examples: list[Example],
    tokenizer: Any,
    device: str,
    cfg: TrainConfig,
    sample: int | None = None,
) -> dict[str, float]:
    """Generate on held-out documents and score the OUTPUT, not the target.

    This is the number that should be read at each checkpoint. It answers the only
    question that matters -- can the model produce a correct document on its own
    -- whereas teacher-forced accuracy answers a question about a model that is
    being fed the answer.

    Reported per checkpoint:

    - `json_rate`      output parses as a JSON object (the gate's C1)
    - `terminated_rate` output emitted the `<|end|>` answer terminator. A model
      that never terminates cannot produce a parseable document by construction,
      so this separates "wrong values" from "cannot finish a sentence".
    - `canonical_rate` only known field names were used (the gate's C2)
    - `nodup_rate`     no key emitted twice (the gate's C3). Duplicates vanish on
      parse, so this reads the raw text.
    - `field_recall`   mean fraction of truth fields extracted correctly
    - `exact_match`    fraction of documents with every field correct

    Scoring goes through `acceptance.py` rather than local copies. An earlier
    version of this file grew its own `_parse_json` while the gate had
    `parse_strict` and the evaluator had `parse_generation`; three parsers that
    had to agree and no mechanism to keep them agreeing.
    """
    from iso20022_lab.model.tokenizer import END_ID, prompt_for

    # Where the model actually is, not where the caller says it is. A caller that
    # passes the wrong device gets a device-mismatch crash from inside
    # nn.Embedding rather than a clear error here, so the argument is ignored.
    device = model_device(model)

    n = cfg.eval_generate if sample is None else sample
    if n <= 0 or not examples:
        return {}

    was_training = model.training
    model.eval()
    subset = stratified_sample(examples, n) if n else []

    parsed_ok = terminated = canonical = nodup = 0
    recalls: list[float] = []
    exacts: list[float] = []

    for item in subset:
        prompt_ids = tokenizer.encode(prompt_for(item["text"]), add_special_tokens=False)
        if len(prompt_ids) >= cfg.max_seq_len:
            continue
        input_ids = torch.tensor([prompt_ids], device=device)
        generated = model.generate_greedy(
            input_ids,
            max_new_tokens=min(cfg.eval_max_new_tokens, cfg.max_seq_len - len(prompt_ids)),
            eos_token_id=END_ID,
            end_token_id=END_ID,
        )
        produced = tokenizer.decode(generated[0, len(prompt_ids) :].tolist())

        if "<|end|>" in produced:
            terminated += 1
        payload, _ = parse_strict(produced)
        if payload:
            parsed_ok += 1
            if set(payload) <= CANONICAL_KEYS:
                canonical += 1
        else:
            # A failed parse still had *some* intent; ``parse_strict`` returning
            # ``{}`` would otherwise silently score 0 and hide which failure it was.
            pass
        if count_duplicate_keys(produced) == 0:
            nodup += 1

        truth = item["values"]
        hits = sum(1 for k, v in truth.items() if payload.get(k) == v)
        recall = hits / len(truth) if truth else 0.0
        recalls.append(recall)
        exacts.append(1.0 if truth and recall == 1.0 else 0.0)

    if was_training:
        model.train()

    docs = len(recalls)
    if docs == 0:
        return {}
    return {
        "docs": float(docs),
        "json_rate": parsed_ok / docs,
        "terminated_rate": terminated / docs,
        "canonical_rate": canonical / docs,
        "nodup_rate": nodup / docs,
        "field_recall": sum(recalls) / docs,
        "exact_match": sum(exacts) / docs,
    }


def move_to_device(model: ISO20022ForCausalLM, device: str) -> ISO20022ForCausalLM:
    """Move the model onto `device`.

    Three ways of writing this were tried and rejected before this one:
    `model.to(device)` and `model.cuda()` both lose their method binding through
    PreTrainedModel's generic bases, so an analyser reports the device argument
    against `self` (or `self` as missing) rather than resolving the call.
    Calling unbound on `nn.Module` is unambiguous and is valid Python.
    """
    if device == "cpu":
        nn.Module.cpu(model)
    else:
        nn.Module.cuda(model)
    return model


def enable_checkpointing(model: ISO20022ForCausalLM, enabled: bool) -> None:
    """Trade compute for activation memory by recomputing layers in backward.

    Set on the body, where the per-layer loop lives. At batch 4 and seq 384 the
    run already peaks at 2.02 GB of a 4 GB card, so recomputation is what buys
    headroom for a larger effective batch rather than the other way round.
    """
    model.model.gradient_checkpointing = enabled


def train_stage1(
    model: ISO20022ForCausalLM,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    cfg: TrainConfig,
    out_dir: Path,
    examples: list[Example] | None = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Teacher-forced extraction training.

    `examples` and `tokenizer` exist for the generation-based evaluation and are
    optional so the function still runs without them. They are passed rather than
    reached for because `train_loader` yields tokenised tensors with the prompt
    masked, and generation needs the *original documents* and the tokenizer to
    build a prompt and decode an answer -- the val_loader cannot supply either.
    """
    # A device string, not torch.device: `.to()` accepts a string, and torch.device
    # is one of the names unresolved through torch._C's wildcard, so naming the
    # type explicitly only moves the error downstream into the `.to()` overload.
    device = cfg.device if torch.cuda.is_available() else "cpu"
    move_to_device(model, device)
    enable_checkpointing(model, cfg.gradient_checkpointing)
    model.train()

    # No weight decay on norms, biases or the attention gate: decaying a
    # normalisation scale toward zero is not regularisation, it just fights the
    # layer's job.
    decay: list[torch.Tensor] = []
    no_decay: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "norm" in name or "gate" in name or "probes" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    optimiser = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
    )

    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_frac))

    history: list[dict[str, Any]] = []
    step = 0
    running = 0.0
    seen_batches = 0
    started = time.time()
    optimiser.zero_grad(set_to_none=True)

    for epoch in range(cfg.epochs):
        for batch_index, (ids, labels) in enumerate(train_loader):
            ids = ids.to(device)
            labels = labels.to(device)
            out = model(
                input_ids=ids,
                labels=labels,
                confidence_weight=cfg.confidence_weight,
            )
            loss = out["loss"]
            assert loss is not None
            (loss / cfg.grad_accum).backward()
            # `.detach()` because `loss` still carries the autograd graph. Converting
            # it with `float()` alone works but PyTorch warns that it "may lead to
            # unexpected behavior" -- and the warning is right in principle: it is
            # the same class of mistake as accumulating a loss tensor and holding
            # every graph in the run. Detaching makes the intent explicit and keeps
            # the log line free of a third of the smoke-test output.
            running += float(loss.detach())
            seen_batches += 1

            is_last = batch_index == len(train_loader) - 1
            if (batch_index + 1) % cfg.grad_accum == 0 or is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                lr = cosine_schedule(
                    step, total_steps, warmup_steps, cfg.learning_rate, cfg.min_lr_frac
                )
                for group in optimiser.param_groups:
                    group["lr"] = lr
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                step += 1

                if step % cfg.log_every == 0:
                    elapsed = time.time() - started
                    print(
                        f"  epoch {epoch + 1} step {step}/{total_steps} "
                        f"loss {running / max(1, seen_batches):.4f} lr {lr:.2e} "
                        f"{elapsed:.0f}s",
                        flush=True,
                    )
                    running = 0.0
                    seen_batches = 0

                if step % cfg.eval_every == 0:
                    metrics = evaluate(model, val_loader, device)
                    metrics["step"] = step
                    metrics["epoch"] = epoch + 1
                    # Generation-based metrics, on the *output*, alongside the
                    # teacher-forced loss. Section 18.10 is the reason: token
                    # accuracy reached 82.14% on a model that extracted 1 field in
                    # 120, because teacher forcing hands the model the correct
                    # prefix. Loss is kept for divergence detection; these are the
                    # numbers that say whether the artifact works.
                    #
                    # This call was written and then *not connected* on the first
                    # attempt -- `generation_metrics` existed, and the log still
                    # printed `token_acc`. Which is the same failure in miniature:
                    # the measurement that matters is the one that runs every time,
                    # not the one that exists.
                    gen = (
                        generation_metrics(model, examples, tokenizer, device, cfg)
                        if examples and tokenizer is not None
                        else {}
                    )
                    metrics.update(gen)
                    history.append(metrics)
                    if gen:
                        print(
                            f"    EVAL step {step}: loss {metrics['loss']:.4f} "
                            f"json {gen['json_rate'] * 100:.0f}% "
                            f"term {gen['terminated_rate'] * 100:.0f}% "
                            f"keys {gen['canonical_rate'] * 100:.0f}% "
                            f"nodup {gen['nodup_rate'] * 100:.0f}% "
                            f"recall {gen['field_recall'] * 100:.1f}% "
                            f"stp {gen['exact_match'] * 100:.1f}%",
                            flush=True,
                        )
                    else:
                        print(
                            f"    EVAL step {step}: loss {metrics['loss']:.4f} "
                            f"token_acc {metrics['token_accuracy'] * 100:.2f}% "
                            "(generation metrics off)",
                            flush=True,
                        )
                    model.train()

                if step % cfg.save_every == 0:
                    saved_path = out_dir / f"step{step}"
                    save_checkpoint(model, saved_path, cfg)
                    _verify_saved(saved_path, model, examples, tokenizer, cfg)

    return {"steps": step, "history": history, "seconds": time.time() - started}


def calibrate_confidence(
    model: ISO20022ForCausalLM,
    examples: list[Example],
    tokenizer: Any,
    cfg: TrainConfig,
    out_dir: Path,
    sample: int = 400,
    bias_correction: bool = True,
) -> dict[str, float]:
    """Stage 2: fit the confidence head to actual correctness.

    Generates an answer per example, scores it against the truth fields, and
    trains the head on that binary outcome. Also reports the reliability of the
    resulting score, because a head that outputs a constant would otherwise look
    like it converged.
    """
    if model.confidence_head is None:
        return {"skipped": 1.0}

    device = cfg.device if torch.cuda.is_available() else "cpu"
    move_to_device(model, device)
    model.eval()

    subset = stratified_sample(examples, sample)
    hidden_states: list[torch.Tensor] = []
    outcomes: list[float] = []
    field_recall: list[float] = []

    # Only generation and feature extraction are no-grad. The head fit below must
    # keep autograd on, so the decorator that used to cover this whole function
    # was wrong: it left `loss.backward()` with nothing to differentiate.
    with torch.no_grad():
        for item in subset:
            prompt_ids = tokenizer.encode(
                prompt_for(item["text"]), add_special_tokens=False
            )
            if len(prompt_ids) >= cfg.max_seq_len:
                continue
            input_ids = torch.tensor([prompt_ids], device=device)
            generated = model.generate_greedy(
                input_ids, max_new_tokens=cfg.max_seq_len - len(prompt_ids)
            )
            produced = tokenizer.decode(generated[0, len(prompt_ids) :].tolist())
            # `parse_strict` from `fields`, not a local parser. This file had its
            # own `_parse_json` until it was removed here: three parsers had grown
            # up independently (this one, `acceptance.parse_strict` and
            # `evaluate_model.parse_generation`) and nothing kept them in
            # agreement. Worse, this copy used the `find("{")`..`rfind("}")`
            # slicing that swept trailing noise INTO the body, so a generation
            # that answered correctly and then kept talking was scored as a parse
            # failure. The shared version scans braces.
            predicted, _why = parse_strict(produced)
            truth = item["values"]
            hits = sum(1 for k, v in truth.items() if predicted.get(k) == v)
            recall = hits / len(truth) if truth else 0.0
            field_recall.append(recall)
            outcomes.append(1.0 if recall == 1.0 else 0.0)

            hidden = model.model(input_ids)
            hidden_states.append(hidden[:, -1, :].detach())

    if not hidden_states:
        return {"skipped": 1.0}

    features = torch.cat(hidden_states, dim=0)
    targets = torch.tensor(outcomes, device=features.device)
    # Exact-match is the label, so a head that always fires is only possible if
    # the model is always right; report the base rate so that is visible.
    base_rate = float(targets.mean())

    head = model.confidence_head
    # The features are detached, so the graph the head builds starts fresh here.
    # `head.proj` and `head.probes` are what receive gradient.
    for param in head.parameters():
        param.requires_grad_(True)
    optimiser = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.0)

    for _ in range(200):
        logits = head(features.unsqueeze(1))
        weight = None
        if bias_correction and 0.0 < base_rate < 1.0:
            # Correct for the class imbalance: exact-match is rare early on, so
            # an uncorrected head minimises loss by predicting "wrong" always.
            positives = targets.sum().clamp(min=1.0)
            negatives = (1.0 - targets).sum().clamp(min=1.0)
            weight = torch.where(
                targets > 0.5, negatives / positives, torch.ones_like(targets)
            )
        loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weight)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        scores = torch.sigmoid(head(features.unsqueeze(1)))
        # Separation between right and wrong answers: a constant head scores 0.
        right = scores[targets > 0.5]
        wrong = scores[targets <= 0.5]
        separation = (
            float(right.mean() - wrong.mean()) if right.numel() and wrong.numel() else 0.0
        )

    save_checkpoint(model, out_dir / "final", cfg)
    return {
        "samples": float(len(targets)),
        "exact_match_rate": base_rate,
        "mean_field_recall": sum(field_recall) / len(field_recall),
        "confidence_separation": separation,
        "mean_score_right": float(right.mean()) if right.numel() else 0.0,
        "mean_score_wrong": float(wrong.mean()) if wrong.numel() else 0.0,
    }


def save_checkpoint(
    model: ISO20022ForCausalLM, path: Path, cfg: TrainConfig | None = None
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    if cfg is not None:
        (path / "train_config.json").write_text(
            json.dumps(cfg.to_json(), indent=2), encoding="utf-8"
        )


def evaluate_checkpoint(
    path: Path,
    examples: list[Example],
    tokenizer: Any,
    cfg: TrainConfig,
    device: str = "cpu",
) -> dict[str, float]:
    """Measure a checkpoint **on disk**, not the in-memory model it came from.

    The log's numbers are produced by the live model. The artifact is a distinct
    object the moment ``save_pretrained`` touches disk, and nothing else in this
    file compared the two, so a run could report 100% recall for a checkpoint
    that generates repetition loops and the report was the only evidence anyone
    read. Loading what was actually written and measuring *that* is the primitive
    that closes the gap.

    ``device`` defaults to CPU on purpose: verification usually runs while
    something else holds the GPU.
    """
    model = ISO20022ForCausalLM.from_pretrained(path)
    model.eval()
    return generation_metrics(model, examples, tokenizer, device, cfg)


def _verify_saved(
    path: Path,
    live: ISO20022ForCausalLM,
    examples: list[Example] | None,
    tokenizer: Any,
    cfg: TrainConfig,
) -> bool:
    """Re-measure a checkpoint just written and compare it to the live model.

    Returns True when they agree. A disagreement is structural -- the log's
    numbers do not describe the file on disk -- so it is printed loudly rather
    than folded in as one more line of routine output.
    """
    if not examples or tokenizer is None:
        return True

    saved = evaluate_checkpoint(path, examples, tokenizer, cfg)
    # No mode handling here: `generation_metrics` saves the training flag, calls
    # `.eval()` and restores it, so doing it again around the call would only
    # duplicate that and give this function a `model` to look up that it has
    # never had -- `live` is the parameter.
    live_metrics = generation_metrics(live, examples, tokenizer, model_device(live), cfg)

    pairs = [
        ("json", saved.get("json_rate", 0.0), live_metrics.get("json_rate", 0.0)),
        ("recall", saved.get("field_recall", 0.0), live_metrics.get("field_recall", 0.0)),
        ("stp", saved.get("exact_match", 0.0), live_metrics.get("exact_match", 0.0)),
    ]
    agree = all(abs(a - b) < 1e-6 for _, a, b in pairs)
    shown = "  ".join(f"{name} {a * 100:.1f}%" for name, a, _ in pairs)
    live_shown = "  ".join(f"{name} {b * 100:.1f}%" for name, _, b in pairs)
    print(
        f"    SAVED {path.name}: {shown}   (live: {live_shown})   "
        f"{'MATCH' if agree else 'MISMATCH'}",
        flush=True,
    )
    if not agree:
        print(
            "    ^ the checkpoint on disk and the live model disagree: the numbers "
            "above the SAVED line do not describe this file.",
            flush=True,
        )
    return agree


def main() -> int:
    parser = argparse.ArgumentParser(description="train the ISO 20022 model")
    parser.add_argument("--docs", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--levels",
        type=int,
        choices=[1, 3],
        default=3,
        help="difficulty levels each value-set is rendered at. 3 (default) "
        "shows every value-set clean, messy and hostile; 1 gives each value-set "
        "one level so the same document budget carries 3x the distinct "
        "value-sets. Generalisation is bounded by distinct value-sets, not "
        "documents.",
    )
    parser.add_argument(
        "--holdout-seed",
        type=int,
        default=None,
        help="value-draw seed for the generation metric's validation corpus. "
        "Defaults to a draw disjoint from --seed, because a held-out slice of the "
        "training draw measures recall of value instances rather than extraction: "
        "the same model scores 97.5% on the within-draw slice and 61.2% on a "
        "fresh draw.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--verify",
        type=Path,
        default=None,
        help="measure a saved checkpoint directory and print what it actually does",
    )
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both")
    args = parser.parse_args()

    tokenizer_path = data_path("tokenizer")
    if not (tokenizer_path / "tokenizer.json").exists():
        print(f"tokenizer missing at {tokenizer_path}; run iso20022_lab.model.tokenizer")
        return 1
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    cfg = TrainConfig(
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        epochs=args.epochs,
        seed=args.seed,
    )

    print("=== building data ===", flush=True)
    examples = build_examples(args.docs, seed=args.seed, levels=args.levels)
    train_examples, val_examples = split_examples(examples, seed=args.seed)
    print(f"  {len(train_examples)} train / {len(val_examples)} val documents")

    # The loader's validation slice checks for divergence during training. The
    # generation metric needs a different corpus entirely: a held-out slice of the
    # training draw sits inside the region the model has already seen, so it
    # reports recall of value instances rather than extraction. Measured on the
    # same checkpoint, the within-draw slice gives 97.5% field recall and a fresh
    # draw gives 61.2%. Reporting the former as generalisation was the error.
    holdout_seed = (
        args.holdout_seed if args.holdout_seed is not None else args.seed + 1_000_000
    )
    if holdout_seed == args.seed:
        print(
            "  --holdout-seed equals --seed, so the generation metric would score "
            "value-sets the model trained on and overstate it. Refusing."
        )
        return 2
    metric_examples = holdout_examples(max(20, args.docs // 4), seed=holdout_seed)
    print(
        f"  generation metric validates on a DISJOINT draw: "
        f"{len(metric_examples)} documents from --holdout-seed {holdout_seed}"
    )

    if args.verify is not None:
        print()
        print(f"=== verifying {args.verify} (measured on the artifact) ===", flush=True)
        metrics = evaluate_checkpoint(args.verify, metric_examples, tokenizer, cfg)
        for key in (
            "json_rate",
            "terminated_rate",
            "canonical_rate",
            "nodup_rate",
            "field_recall",
            "exact_match",
        ):
            if key in metrics:
                print(f"  {key:<18} {metrics[key] * 100:6.2f}%")
        return 0

    train_set = ExtractionDataset(train_examples, tokenizer, cfg.max_seq_len)
    val_set = ExtractionDataset(val_examples, tokenizer, cfg.max_seq_len)
    print(f"  tokenised: {len(train_set)} train ({train_set.skipped} skipped)")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate
    )

    config = ISO20022Config(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=PAD_ID,
        bos_token_id=1,
        eos_token_id=EOS_ID,
    )
    declared = config.parameter_count()
    print()
    print("=== model ===")
    print(f"  declared params: {declared['total']:,}")
    print(f"  engram: {declared['engram']:,} over layers {config.engram_layers}")
    print(f"  confidence head: {declared['confidence_head']:,}")

    if args.resume:
        print(f"  resuming from {args.resume}")
        model = ISO20022ForCausalLM.from_pretrained(args.resume)
    else:
        model = ISO20022ForCausalLM(config)
    live = parameter_count(model)
    print(f"  live params: {live:,}  MATCH={live == declared['total']}")

    result: dict[str, Any] = {}
    if args.stage in ("1", "both"):
        print()
        print("=== stage 1: extraction ===", flush=True)
        result["stage1"] = train_stage1(
            model,
            train_loader,
            val_loader,
            cfg,
            args.out,
            examples=metric_examples,
            tokenizer=tokenizer,
        )
        # Recorded here because this is where the counts are known. The model card
        # reads `train_docs`, `val_docs` and `sequences` to state what the model
        # was trained on, and reported 0 for all three when they were absent --
        # a published card claiming the model saw no data at all.
        result["stage1"].update(
            train_docs=len(train_examples),
            val_docs=len(val_examples),
            sequences=len(train_set),
        )
        save_checkpoint(model, args.out / "final", cfg)

    if args.stage in ("2", "both"):
        print()
        print("=== stage 2: confidence calibration ===", flush=True)
        result["stage2"] = calibrate_confidence(
            model, val_examples, tokenizer, cfg, args.out
        )
        for key, value in result["stage2"].items():
            print(f"  {key:<24} {value:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train_report.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print()
    print(f"saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
