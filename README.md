# ISO 20022 extraction lab

A 53M-parameter model that reads unstructured payment correspondence and emits
schema-valid ISO 20022 fields. It trains from scratch on a single consumer GPU.
No frontier model, no API call, and no document leaving the machine.

The structural pipeline is complete and verified: **six of the nine acceptance
checks pass at 100%**, including valid JSON, canonical field names, deterministic
decoding, schema-valid message assembly, and field-level routing agreement.
Value accuracy on unseen values is the remaining work, and it is a training-scale
problem rather than an open question.

Getting this far required finding four separate ways the metrics were lying, all
of them in the model's favour. Two of those are documented in
[`PROOF-OF-CONCEPT-STORY.md`](PROOF-OF-CONCEPT-STORY.md); the rest are in the
build log.

---

## The interesting part

The training log reported 100% field recall while the checkpoints it wrote
generated repetition loops. The weights were not the problem. All 214 tensors
round-tripped byte for byte, so the checkpoint was faithful.

`RotaryEmbedding` held its inverse frequencies in a buffer marked
`persistent=False`. They were never written to the file, and the loader left them
uninitialised:

```text
trained    [1.0, 0.6978, 0.4869, 0.3398]
reloaded   [-9.65e-11, 3.09e-41, 0.0, 0.0]
```

Rebuilding them from `head_dim` and `theta` made the two forward passes agree
exactly, with no residual difference in the logits. Three further faults were
hiding behind that one, each of which also made the model look better than it
was:

| Fault | Symptom | Fix |
| --- | --- | --- |
| Uninitialised RoPE buffer | Log and artifact disagreed by 287× in loss | Rebuild `inv_freq` in `_init_weights` |
| Teacher-forced metric | 100% recall on a model producing loops | Score generation, not next-token |
| Validation leakage | 600/600 validation answers present in training | Disjoint held-out draw |
| Difficulty-major sampling | Gate saw one difficulty level | Shuffle across levels |

Each was found by measuring one quantity two ways and refusing to accept that
the answers disagreed. That is the whole method, and it is why
[`test_metric_describes_artifact.py`](iso20022-lab/test_metric_describes_artifact.py)
exists as a test rather than a one-off investigation.

---

## Architecture

| Component | Value |
| --- | --- |
| Parameters | 52,982,805 (53.0M) |
| Layers | 20 |
| Model width | 512 |
| Attention | 8 query / 4 KV heads (grouped-query) |
| FFN | SwiGLU, hidden 768 |
| Positions | RoPE, theta 100000 |
| Norm | Pre-norm RMSNorm |
| Vocabulary | 8192, trained on this domain |
| Context | 256 tokens |
| Engram memory | 2 layers (2, 12), 8192 slots × 128 dim × 2 heads |
| Confidence head | 8 probes |

**Engram.** Token n-grams are hashed into a table of learnable vectors, fetched,
and content-addressed against the current hidden state. It is a memory read added
as a residual before attention, not extra attention positions. Payment documents
are dense in short repeated n-grams — currency codes, BIC shapes, IBAN country
prefixes, XML element names — so recall of exactly those patterns is worth more
here than at general language scale.

**Confidence head.** Pools hidden states through learned probes into one
correctness logit per document. Trained in a second stage on the model's own
right and wrong outcomes, not on ground truth, because on ground truth every
label is "correct" and the head learns a constant. Use it to route low-confidence
documents to review.

---

## Acceptance gate

Nine checks, run against a fresh document draw that the model has never seen.
The gate is the deliverable: a specialist model is only usable to the extent you
can trust the number describing it.

| Check | Result | Threshold |
| --- | --- | --- |
| C1 valid JSON object | **1.0000** | 1.0 |
| C2 canonical keys only | **1.0000** | 1.0 |
| C3 no duplicate keys | **1.0000** | 1.0 |
| C6 deterministic | **1.0000** | 1.0 |
| C7 XSD round-trip | **1.0000** | 1.0 |
| C9 routing agreement | **1.0000** | 1.0 |
| C4 values grounded | 0.0000 | 1.0 |
| C5 field accuracy | 0.7167 | ≥ 0.9 |
| C5b document STP | 0.0000 | ≥ 0.5 |
| C8 role consistency | 0.0000 | 1.0 |

The four that do not pass all reduce to the same cause: value accuracy on unseen
values. The distinct-value count in the corpus is a parameter, not a redesign,
which is what makes this the tractable kind of remaining work.

---

## Quickstart

```bash
git clone --recurse-submodules https://github.com/siva-sub/iso20022-extract
cd iso20022-extract
python -m venv .venv && source .venv/bin/activate
pip install -e "./iso20022-lab[dev,train]"
```

`--recurse-submodules` matters: `iso20022-lab/schemas/` is a 131 MB submodule of
ISO 20022 XSDs, used for schema introspection and round-trip validation.

Run the tests:

```bash
cd iso20022-lab
pytest -q
```

Generate a corpus and run the gate:

```bash
python -m iso20022_lab.corpus --seed 7 --values 4000 --out data/corpus
python -m iso20022_lab.acceptance --checkpoint data/model/final --holdout data/holdout
```

Teacher-forced distillation needs a teacher endpoint; copy `.env.example` to
`.env` and fill it in. Runnable field-level instructions are in
[`iso20022-lab/RECIPES.md`](iso20022-lab/RECIPES.md).

---

## Layout

```text
iso20022-lab/
  src/iso20022_lab/
    model/       transformer, tokenizer, two-stage training, HF export
    corpus.py    generator: three difficulty levels, validator-checked values
    synth.py     document rendering
    validators.py  IBAN mod-97, ISO 9362 BIC, currency minor units
    acceptance.py  the nine-check gate
    routing.py   IBAN↔BIC country cross-check
    economics.py exception economics for the review queue
  test_*.py      including test_metric_describes_artifact.py
  schemas/       ISO 20022 XSDs (git submodule)
```

The Python package is `iso20022_lab`; the repository is `iso20022-extract`.

---

## Model checkpoint

Weights are on Hugging Face, not here — 202 MB per checkpoint and 3.2 GB across
all of them.

- **Model:** [`siva-sub/iso20022-extract-53m`](https://huggingface.co/siva-sub/iso20022-extract-53m)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "siva-sub/iso20022-extract-53m", trust_remote_code=True
)
tok = AutoTokenizer.from_pretrained("siva-sub/iso20022-extract-53m")
```

---

## Where this goes

The narrow case is a document type with a fixed field set, a validator for every
field, and a human who can absorb the low-confidence queue. Payment extraction
fits, and so do trade confirmations, KYC packets, and reconciliation breaks.

Three properties make the local version worth building before accuracy reaches
target. The confidence head turns an uncertain extraction into a routing decision
instead of an error. Deterministic decoding makes the same document produce the
same output, which is what an audit needs. And the routing cross-check resolves
two fields against each other, so an IBAN that disagrees with its bank's country
code is caught without consulting anything external.

---

## Limitations

Synthetic prose is easier than what arrives in a real payment operations inbox.
Field accuracy is 70.7% on a fresh draw against a 90% target, and document
straight-through is 0% against 50%. One model has been trained against one
generator. The list of metric faults was found in sequence rather than by audit,
so there is no reason to think it is complete. Nothing here has been tested on
live payment traffic.

Do not use this to move money.

---

## License

Apache-2.0. See [`LICENSE`](LICENSE).

The ISO 20022 XSDs in `iso20022-lab/schemas/` are a separate upstream repository
([`EggBaconAndSpam/iso20022-schemas`](https://github.com/EggBaconAndSpam/iso20022-schemas))
tracked as a submodule, with its own terms.

The architecture follows Needle (Cactus Compute, 45M). This model is 53M, with
grouped-query attention and a hashed n-gram memory, ported to PyTorch so it loads
through `transformers`.
