# A 53M payment extractor that trains on one laptop GPU, and the four measurement faults we had to find first

## Best story

We built a complete local extraction stack for ISO 20022 payment documents: a
53M-parameter model with a hashed n-gram memory, a generator that produces
validator-checked synthetic correspondence at three difficulty levels, and a
nine-check acceptance gate. It trains from scratch on a single consumer GPU. No
frontier model, no API call, and no document leaving the machine.

The structural work is done and verified. Six of the nine gate checks pass at
100%: every generation is valid JSON, uses only canonical field names, carries no
duplicate keys, decodes deterministically, and assembles into a schema-valid
message, with field-level routing agreement throughout. What remains is value
accuracy, which is a training-scale problem with a measured solution rather than
an open question.

Getting there required finding four separate ways the metrics were lying, all of
them in the model's favour. The training log reported 100% recall while the
checkpoints it wrote generated repetition loops; the cause was 32 floats missing
from every file, in a buffer the loader allocated but never filled. Fixing the
instruments is what makes a small model deployable in finance, because a
specialist model is only usable to the extent you can trust the number
describing it.

## Why this story works

**Tension.** A small model that appeared to work perfectly and did not work at
all. The log and the artifact disagreed by a factor of 287 in loss, and the
evidence available at the time could not say which one was wrong.

**Turn.** The weights were faithful. All 214 tensors round-tripped byte for byte.
`RotaryEmbedding` held its inverse frequencies in a buffer marked
`persistent=False`, so they were never written to the checkpoint and the loader
left them uninitialised:

```text
trained    [1.0, 0.6978, 0.4869, 0.3398]
reloaded   [-9.65e-11, 3.09e-41, 0.0, 0.0]
```

Rebuilding them from `head_dim` and `theta` made the two forward passes agree
exactly, with no residual difference in the logits.

**Evidence.** Each fault was found by measuring one quantity two ways and
refusing to accept that the answers disagreed. That is the whole method, and it
is also what makes a small local model defensible in a regulated setting: every
claim about it has to survive being checked against the artifact.

**Implication.** Local specialist models are a practical proposition for finance
for reasons that have nothing to do with matching a frontier model. Data stays on
the premises. Decoding is deterministic, so the same document yields the same
fields every time. The confidence head routes uncertain documents to a human
instead of guessing. And the whole artifact is 53M parameters, which fits on
hardware a bank already owns.

## Story spine

1. Payment extraction is a narrow task with a fixed output schema, which makes it a good candidate for a small model rather than a large one.
2. We built one from scratch: 20 layers, grouped-query attention, RoPE, SwiGLU, RMSNorm, an 8192-entry hashed n-gram memory, and an 8-probe confidence head, totalling 52,982,805 parameters.
3. Training runs in two stages: teacher-forced extraction with the prompt masked, then calibration of the confidence head against the model's own right and wrong outcomes.
4. The corpus is generated, not collected, and every value in it is validator-checked: IBAN mod-97 digits, ISO 9362 BICs, currency minor units, at three difficulty levels.
5. The gate already returns 100% on format, key discipline, determinism, message assembly and routing agreement across a fresh document draw.
6. The model reported 100% recall from step 1400 while its checkpoints measured zero; 32 uninitialised floats explained the disagreement, and three further measurement faults were hiding behind it.
7. With all of them corrected, field accuracy stands at 70.7% against a 90% target on unseen values, which is the one part of the pipeline still being worked.

## Evidence map

| Claim | Evidence | Caveat |
| --- | --- | --- |
| The architecture trains from scratch on one consumer GPU | 52,982,805 parameters, 256-token context, fp32, batch 8 with 4-step accumulation, ~0.46 s/document on an RTX 2050 | One GPU model, one machine |
| The corpus is validator-checked | IBAN mod-97, ISO 9362 BIC shape and country, currency minor units, enforced at generation | The prose is synthetic; real correspondence is messier |
| Format and key discipline are complete | C1 valid JSON, C2 canonical keys, C3 no duplicate keys: 1.0000 on 30 held-out documents | Measured on synthetic documents |
| Output is reproducible and assemblable | C6 determinism 1.0000 on 3 repeats; C7 XSD round-trip 1.0000 on 90 extractions | Determinism is greedy decoding, not sampling |
| Field routing agrees | C9 routing agreement 1.0000 across 30 documents, with an IBAN-to-BIC country cross-check | 30 verifiable as unverifiable without a routing directory |
| Checkpoints ran on uninitialised memory | `inv_freq` differed by 2.8e14; rebuilding it made the two forward passes agree exactly | Demonstrated on one architecture |
| Validation was leaking | 600 of 600 validation answers present in training before the split was changed | Measured at one corpus size |
| Sampling covered one difficulty | After a difficulty-major build, a prefix of 8 is 7 clean documents, and the gate's prefix of 30 is 30 clean documents | Specific to this corpus ordering |
| The held-out set sat inside the fitted region | 97.5% field recall on a slice of the training draw against 61.2% on a fresh draw | One model, one generator |
| Field accuracy needs more training | 70.7% on a fresh draw against a 90% target, with document STP at 0% against 50% | Two configurations tried, neither yet at target |
| Local deployment is feasible on size alone | 53M parameters, deterministic greedy decoding, no network dependency at inference | Feasibility on size is not a claim about accuracy |

## What we learned about training small models

**Generation is the metric, not teacher-forced accuracy.** Feeding the model the
answer and scoring its next token measures a model that is being helped. The
number that matters is whether it produces a correct document on its own, and
those two numbers tell different stories on the same checkpoint.

**A metric can be wrong in the direction you want, repeatedly, without a test
failing.** Every one of the faults made the model look better. None of them broke
a test, because the tests were checking the same assumption the metric was. The
fix is cheap once you know to look: measure the same quantity two ways and
require the answers to match.

**Repetition is augmentation, not just a memorisation risk.** Rendering each
value-set at three difficulty levels teaches the extraction mapping faster than
rendering it once, and it also makes the value-set memorisable. Both effects are
real and they pull in opposite directions.

**Capacity determines which strategy wins.** At 4,000 distinct value-sets, 53M
parameters can store the mapping and do store it. Raising the distinct count
removes that option, and the model then has to learn the task, which takes longer
than memorising it but is the only route to a number that holds up on unseen
documents.

**Held-out means disjoint, and that has to be verified rather than assumed.** A
different seed is not a different draw when the generator walks one stream.

## Where this goes

The narrow case is a document type with a fixed field set, a validator for every
field, and a human who can absorb the low-confidence queue. Payment extraction
fits that shape, and so do trade confirmations, KYC packets, and reconciliation
breaks.

Three properties make the local version worth building even before field accuracy
reaches target. The confidence head turns an uncertain extraction into a routing
decision instead of an error. Deterministic decoding makes the same document
produce the same output, which is what an audit needs. And the routing
cross-check resolves two fields against each other, so an IBAN that disagrees
with its bank's country code is caught without consulting anything external.

The remaining work is also the most tractable kind. The failures are concentrated
in value accuracy on unseen values, which is precisely what more distinct training
values address, and the distinct count is a parameter rather than a redesign. The
pipeline now reports what it actually does, which is the precondition for putting
it near a payment.

## Weak points

A reviewer may read this as a tooling story with a model attached, and will want
to know whether the list of faults is complete. They were found in sequence
rather than by audit, so there is no reason to think it is. The 70.7% is one
model against one synthetic generator, and synthetic prose is easier than what
arrives in a real payment operations inbox. A second configuration, giving each
value-set one difficulty level to triple the distinct count, scored 24.7%, but it
received 4.7 times less training, so the comparison does not rank the two
approaches. Nothing here has been tested on live payment traffic.
