# Extracting ISO 20022 messages from unstructured documents

**Reader:** you, in three weeks, having forgotten why any of this is shaped the way it is.
**After reading you can:** build and train the extractor, defend the design on economic grounds, and know which parts are proven versus assumed.

**Status:** Section 2 is a list of things that broke under test. Read it first. The design in section 5 is the revision forced by those failures, not the original plan.

**Deciding whether to build this at all:** read section 17. It is the decision procedure, and it concludes against the model for small, stable format counts.

**Building the front door:** read section 18. Most inbound needs no OCR, and the measure that decides which does is a probe over the output validators.

---

## 1. The economic shape

Payments runs on enormous volume and thin per-item margin. The industry measures **straight-through processing (STP)** — the share of messages that pass end to end without human intervention — and every point of STP lost converts directly into manual work, which is where the margin goes.

Two forces are pushing volume into the manual queue.

**A dated mandate.** SWIFT's CBPR+ programme is migrating cross-border payments from MT to ISO 20022 MX. This is not optional. As of **14 November 2026** SWIFT mandates that institutions be able to receive Enquiry and Investigation messages in MX format (`camt.110` and related). Dated mandates are the strongest demand in enterprise software because the alternative is losing access to a payment rail.

**Lossy mapping.** A legacy MT103 and a `pacs.008` carry the same intent through different structures. Field `:50K:` ordering-customer is ordered text lines; `Dbtr` is a structured party with postal address, identification and scheme. A careless mapping produces messages that are structurally valid and semantically wrong — the worst kind, because they pass validation and fail reconciliation downstream.

So the work is real and dated. The question is what shape the software should take.

---

## 2. What broke under test

Every claim below was tested against the real schemas and the real 45M model. Four of these invalidate parts of the original design.

### 2.1 The architecture core holds

| Test | Result |
|---|---|
| Shuffled slot dict produces byte-identical XML | **HOLDS** |
| Generated message validates against the real XSD | **PASSES** |
| Bad input is caught and names the failing path | **WORKS** |

The central claim is proven: element order comes from the schema, not from the model or from dict ordering. This is the part that survives scrutiny.

### 2.2 A real bug, found and fixed: `xs:choice`

The first run **failed XSD validation** with `Element 'EqvtAmt': This element is not expected.`

Cause: the introspector flattened `xs:choice` into a plain child list, losing the "exactly one of these" semantics. `AmountType4Choice` is a choice of `InstdAmt` *or* `EqvtAmt`; the builder emitted both. The same bug hit `AccountIdentification4Choice` (`IBAN` *or* `Othr`) and `DateAndDateTime2Choice` (`Dt` *or* `DtTm`) — all pervasive in ISO 20022.

It also inflated the required-field count. `pain.001.001.09` reported **362 required leaf paths**; the correct figure is **174**, because choice branches were being counted as individually mandatory.

Fixed by modelling choice explicitly: a parent carries `is_choice`, its branches carry `in_choice`, and the serializer emits exactly one branch. This class of bug is invisible in a demo with one happy path and fatal in production, because every choice type in the schema is a landmine.

### 2.3 The context window kills document-level extraction

This is the most serious finding.

| Input | Result |
|---|---|
| 76 chars (~19 tokens), clean sentence | Correct extraction (though `debtor`/`creditor` were **swapped**) |
| 599 chars (~149 tokens), realistic invoice | **Complete garbage** |

On the 149-token invoice the model extracted `creditor_name: "Unit 4"` (an address line), `creditor_iban: "Bletchley Industrial Estate"` (also an address), and `amount: 150`. Worse, its reasoning invented parameters the tool does not have — `bank_name`, `payment_reference`, `delivery_note`, `receipt_id`, `condition`. The tool defines six arguments.

The engine documents a **256-token sliding window** with tools pinned as KV sinks. Real invoices exceed that comfortably. The frozen base model cannot do document-level extraction, and no amount of prompt tuning fixes a window that small.

### 2.4 The confidence gate does not survive fine-tuning

The package states it directly, and the code confirms it:

```text
finetuning does not update the confidence head, so scores are
uncalibrated for tuned weights; this agent reports confidence as None
response["confidence"] = None
```

So the escalation gate that the entire safety story depends on **disappears the moment you train the model**. The original plan said "fit the confidence threshold to a target escalation rate" while simultaneously planning to LoRA-fine-tune — a direct contradiction.

### 2.5 Grounding catches fabrications — the one thing that worked

When the model invented values, the call was correctly withheld:

| Case | Model produced | Outcome |
| --- | --- | --- |
| No IBAN in text | `credit_iban: 'CRED'` from "Bank details to follow" | `call: []`, confidence 0.0 |
| Amount in words only | `credit_iban: 'GmbH'`, `creditor_name: 'ACME GmbH'` | `call: []`, confidence 0.0 |
| Silent on charge bearer | `creditor_name: 'DE89370400440532013000'` | `call: []`, confidence 0.0 |

The reasoning text hallucinated freely, but the **grounding check rejected the output**. This is real value: the *architecture* protects you when the model fails. Note carefully that the protection came from validation, not from the model being good.

### 2.6 Non-determinism

Near-identical inputs produced different outcomes across sessions — one extracted correctly, a later one refused. Same model, same weights, different answer. For a regulated pipeline this means every path needs a deterministic backstop, and it means "it worked when I tried it" is not evidence.

### 2.7 End token and repetition

The specific failure you asked about — a missing end token causing runaway repetition — behaves as follows.

The **call** is grammar-constrained, so it cannot repeat freely. Tested output was `[]`, two characters, clean termination. The quirk is contained for structured output.

The **reasoning field is generated unconstrained**, per the model documentation. That is where repetition, rambling, and hallucinated parameter names appear — exactly what section 2.3 shows. So the exposure is real but it lives in the one field that should never reach the output document.

### 2.8 ISO 4217 minor units

All three currency cases failed to extract at all: `JPY 125000` (0 decimals), `BHD 1250.125` (3 decimals), `EUR 1250.50` (2 decimals). The model has no concept of minor units. `JPY` with decimals is invalid, and only deterministic code can enforce that.

---

## 3. The economic argument, stress-tested

My own earlier economic case was overstated. Corrections follow.

### 3.1 "Zero marginal cost" is false

Local inference has no per-token bill, but it is not free. You take on engineering time, model versioning, drift monitoring, retraining, evaluation harnesses, and on-call. The honest statement is **near-zero marginal cost and a substantial fixed cost plus a real ownership cost.**

### 3.2 The break-even is a number, and it is not small

Assume a hosted API at $0.05/message (mid-range for a document plus schema prompt), and a local build at roughly $20k of loaded engineering plus $4k/year maintenance:

| Volume | API cost/month | Local |
| --- | --- | --- |
| 10k msgs/month | $500 | Fixed cost never amortises: ~40 months to break even |
| 100k msgs/month | $5,000 | ~4 months to break even |
| 1M msgs/month | $50,000 | ~2 weeks to break even |

Break-even sits near **400,000 messages**. Below roughly 100k/month the API is cheaper once you price your own engineering honestly. The original document implied local always wins. It does not — local wins **above a volume threshold**, and the threshold is high.

### 3.3 "An STP point is worth more than the software budget" was asserted, not measured

It is directionally right and it is exactly the kind of claim a buyer will challenge. It needs a real number from a real process: exceptions per day, minutes per exception, loaded cost per hour. Until that exists, treat it as a hypothesis to validate in Phase 3, not a premise.

### 3.4 The capability claim collapsed

Section 2.3 shows the frozen 45M model failing at document scale. The economics of a local model are irrelevant if it cannot do the job. The economic case now depends entirely on whether a **fine-tuned** small model, given short segmented spans, reaches usable accuracy. That is an open question, and Phase 4 is where it gets answered.

### 3.5 The safety claim collapsed

Section 2.4 removes the confidence gate for tuned weights. An uncalibrated pipeline cannot be safely automated, and an unsafe pipeline has negative economic value — a wrong payment costs far more than a manual review.

### 3.6 What survives

The genuinely strong arguments hold:

- **Data residency removes a real cost centre.** No egress means no DPAs, no transfer mechanisms, no third-party audit rights, no vendor breach exposure. The `vgi-iso20022` maintainers market this for exactly this reason.
- **Latency.** No network round trip matters on instant rails and against payment cut-offs.
- **Auditability.** A 14 MB artifact can be hashed, versioned, and frozen. An API that updates silently cannot be, and regulated change control requires the former.
- **Cost above threshold.** Past roughly 100k messages/month the fixed-cost structure wins decisively.

The correct summary: local inference is an *infrastructure decision with a volume threshold*, not a universal win.

---

## 4. Why a small model, on the merits

Not a compromise. A consequence of where the required knowledge lives.

From *A Controlled Study of Attention-Only Transformers* (arXiv:2607.18363):

> attention-only models are **better on context-grounded answers** and worse where knowledge must come from weights.

Every extraction task decomposes into three kinds of knowledge:

| Kind | Where it lives | Who supplies it | Cost |
| --- | --- | --- |---|
| **Structural** — order, cardinality, namespaces, code lists | The XSD | Deterministic code | Zero, and exact |
| **Contextual** — which value in *this* text is the creditor IBAN | The document | The model | Small |
| **Parametric** — checksums, minor units, scheme rules | Weights, if you let them | Rules | Should be zero |

The schema is in the context. The values are in the document. Almost nothing needs to come from weights, which is why a frontier model bills you for capacity this task does not use.

Section 2.8 is the proof: the model failed at minor units, which is a *parametric* fact. Push it into code and the failure disappears. That is the pattern — every failure in section 2 is either structural (the choice bug) or parametric (minor units, checksums), and both belong in deterministic code.

The residual — genuine language understanding of messy text — is the only part that needs a model, and it is small enough for a small model.

---

## 5. Revised architecture

The failures force three additions the original design lacked: a **segmenter**, an **independent calibration model**, and **deterministic validators carrying correctness**.

```text
  document (too long for the window)
          │
          ▼
  ┌───────────────────────────┐
  │  SEGMENTER                │  window into spans that fit the context
  │  deterministic + model    │  NEW: forced by section 2.3
  └───────────────────────────┘
          │
          ▼
  candidate spans                     short text, bounded length
          │
          ▼
  ┌───────────────────────────┐
  │  SLM: slot extraction     │  grammar-constrained, values only
  └───────────────────────────┘
          │
          ▼
  ┌───────────────────────────┐
  │  DETERMINISTIC VALIDATORS │  IBAN mod-97, BIC, UETR, LEI,
  │                           │  ISO 4217 minor units, code lists
  └───────────────────────────┘  NEW: carries correctness, not the model
          │
          ▼
  ┌───────────────────────────┐
  │  XSD-driven serializer    │  element order from xs:sequence
  │                           │  PROVEN (section 2.1)
  └───────────────────────────┘
          │
          ▼
      ISO 20022 XML ────────► XSD + Schematron + usage guidelines
          │
          ▼
  ┌───────────────────────────┐
  │  CALIBRATION MODEL        │  independent of the extractor
  └───────────────────────────┘  NEW: forced by section 2.4
          │
     auto-approve  |  escalate
```

### 5.1 Segmenter

Because 256 tokens cannot hold an invoice, segmentation is mandatory, not an optimisation. Deterministic anchors first — "IBAN", "BIC", "Total", "Amount due", "Remit to" — then a model pass only for spans the anchors miss. Each span is extracted independently, and the results are merged with provenance back to the source span.

This also improves auditability: every extracted value points at the text it came from.

### 5.2 Deterministic validators

These carry correctness and must never be delegated to the model:

- **IBAN** — mod-97 checksum, country length, structure
- **BIC** — 8 or 11 characters, country and location codes
- **UETR** — UUID shape and version
- **ISO 4217** — currency validity *and minor units* (JPY 0, BHD 3, EUR 2)
- **External code sets** — `ChrgBr`, `Purp`, `CtgyPurp`, `SvcLvl`, `LclInstrm`
- **Amount arithmetic** — line items summing to the stated total

A failure here is a hard reject, not a confidence adjustment.

### 5.3 Calibration, rebuilt

Since the bundled head is disabled for tuned weights, calibration must be independent. Two workable options:

1. **Train a separate small classifier** on `(span, extracted_slots, validator_outcomes)` → correct/incorrect, and calibrate it.
2. **Gate on deterministic signals** — validator pass/fail, span count, whether the extraction was ambiguous, whether the model produced any fabrication attempts.

Option 2 is cheaper, auditable, and needs no training. Start there. Option 1 only if measurement shows it adds value.

---

## 6. The schema stack

The XSD is necessary and not sufficient. Eight artifacts constrain a production message.

| # | Source | What it gives | In the XSD? |
| --- | --- | --- |---|
| 1 | **XSD** (2,641 versions, vendored) | Element tree, cardinality, order, internal code lists | — |
| 2 | **External code sets** | `ChrgBr`, `Purp`, `CtgyPurp`, `SvcLvl`, `LclInstrm` | No — a machine-readable version is published by the ISO 20022 TSG |
| 3 | **Schematron** | Business rules XSD cannot express | No |
| 4 | **MDR** (Message Definition Report) | Authoritative per-element semantics | No |
| 5 | **MUG** (Message Usage Guide) | Intended usage, worked examples | No |
| 6 | **Usage guidelines** — CBPR+, SEPA/EPC, HVPS+, FedNow, T2, CHAPS | The real constraint: a strict subset | No |
| 7 | **ISO 4217** | Currency and minor units | Partly |
| 8 | **IBAN / BIC / LEI registries** | Checksums, shape | No |

**A green XSD validation means structurally sound, not accepted.** CBPR+ is stricter than the base schema, so valid messages still get rejected in production. Item 6 decides STP outcomes and is what the extractor should be conditioned on.

---

## 7. Data: where to actually get it

### 7.1 The paired dataset does not exist, and will not

| Source | What it is | Paired? |
| --- | --- | --- |
| `vivekgupta/ISO20022` (HF) | 490 Q&A rows, 177 kB — prose *about* the standard | No |
| `husseinsalaudeen/iso20022-payments-analytics-synthetic-v1` (Kaggle) | Synthetic messages for analytics | No unstructured side |
| IEEE DataPort synthetic pacs.008/009 | Messages for fraud detection | No unstructured side |

Published data is either prose about the standard or structured messages alone. The unstructured side of a real pair is an invoice containing names, IBANs and amounts. That is PII and commercially sensitive, so it is not open. Searching harder does not change this.

### 7.2 What is actually available and useful

The useful public data is not pairs. It is **document corpora** (to learn real layouts and language), **value distributions** (to sample realistic fields), and **real messages** (to validate the serializer).

#### Document corpora — the unstructured side

| Source | Content | Use |
| --- | --- | --- |
| **DocILE** (`rossumai/docile`, arXiv 2302.05658) | 6.7k annotated + **100k synthetic** + ~1M unlabeled business documents (invoices, orders). KILE and LIR tracks | **The single most valuable source.** 100k synthetic documents with key-information annotations |
| `longmaodata/Invoice-annotation` (HF) | Invoice key-field classification and extraction | Field vocabulary and layout patterns |
| `manuelaschrittwieser/invoice-extraction-dataset-v2` (HF) | Invoice extraction set | Field coverage |
| `mathieu1256/FATURA2-invoices` (HF) | FATURA multi-layout invoice images | Layout diversity |
| `mychen76/invoices-and-receipts_ocr_v1` (HF) | Invoices and receipts with OCR | Text-noise realism |
| `GokulRajaR/invoice-ocr-json`, `shubh303/Invoice-to-Json` (HF) | Invoice → JSON pairs | Direct supervision for the extraction shape |
| `Voxel51/scanned_receipts` (HF) | ICDAR-SROIE, 1,000 real scanned receipts | Real-world noise |
| `Voxel51/consolidated_receipt_dataset` (HF) | 11,000+ Indonesian receipts with OCR boxes | Language diversity |
| `priyank-m/SROIE_2019_text_recognition` (HF) | SROIE text | Text-only variant |
| `Panhapich/bank-statement-detection` (HF) | Bank statement tables | **Closest public analogue to camt.053** |
| Kaggle: `osamahosamabdellatif/high-quality-invoice-images-for-ocr` | Scanned + digital invoices | OCR realism |
| Kaggle: `husseinsalaudeen/iso20022-payments-analytics-synthetic-v1` | Synthetic ISO 20022 messages | Structured-side realism |

#### Value distributions — for realistic sampling

| Source | Content |
|---|---|
| `gretelai/synthetic_pii_finance_multilingual` (HF) | Synthetic financial PII across languages |
| Mendeley, *Synthetic Dataset for PII Detection and Anonymization in Financial Documents* | Synthetic PII for financial documents |

These matter more than they look. The generator needs realistic names, addresses, IBANs and narratives, and these give you the distributions without touching real PII.

#### Real messages and fixtures — to validate the serializer

| Source | Content |
|---|---|
| `EggBaconAndSpam/iso20022-schemas` | 2,641 XSDs (already vendored) |
| `Query-farm/vgi-iso20022` | Committed golden fixtures, plus MT and MX readers |
| `sebastienrousseau/pain001` | Generator/validator with XSD and SEPA validation, Apache-2.0 |
| `issettled/iso20022-issettled` | XSDs plus sample messages and templates |
| `galactixx/iso20022-message-catalogue` | Daily scraper that organises all published schemas |
| GitHub topic `mt103` | MT103 → pacs.008 converters with test messages |
| Goldman Sachs developer sample `camt.053` | A full, realistic statement (verified working) |

#### Generators — for the value layer

Faker-style libraries produce valid IBANs, BICs, LEIs, company names, addresses and currencies in bulk. Combined with the ISO 4217 minor-unit table they cover most field-level realism.

### 7.3 The synthetic-pair pipeline, restated

Given the corpora above, the data plan is:

1. **Sample a valid message** from the XSD (required fields always, optional probabilistically, code-list values only from real enumerations, exactly one branch per choice).
2. **Render it** into unstructured surfaces, borrowing *layout patterns* from DocILE/FATURA and *language* from the invoice corpora, with values drawn from the distributions above.
3. **Emit the pair.** The label is the generated message — correct by construction.
4. **Hold out DocILE's human-annotated 6.7k** and real receipts as an out-of-distribution test set that the renderer never saw.

Step 4 is what keeps the evaluation honest. Without it you measure your own renderer.

### 7.4 The privacy point that makes this the right approach

Synthetic pairs contain **no real PII at all**. That is not a convenience, it is the reason the approach is viable:

- You can train on millions of pairs with no data protection agreement and no lawful basis question.
- You can ship the resulting dataset and the trained model openly.
- You never introduce a retention or breach obligation for data that was never real.

Combined with local inference, this means the entire pipeline can run with **zero egress and zero real personal data in the training loop.** For a regulated institution that is a materially stronger position than any hosted alternative, and it is the strongest surviving economic and legal argument in this document.

One caveat worth stating: local processing relocates liability rather than removing it. You now hold payment data in your own system, so you own the breach exposure and the retention policy. And under GDPR Article 22, a fully automated decision producing legal effect may require human review regardless of how good the confidence score is — which is a further reason the escalation path is not optional.

---

## 8. Training plan

Corrected for everything in section 2.

### 8.1 What to train, and on what

| Stage | Objective | Data |
| --- | --- | --- |
| **Segmenter** | Span boundaries carrying extractable fields | Synthetic documents with known field positions |
| **Slot extractor** | Short span → typed values | Synthetic pairs, chunked to fit the window |
| **Calibration** | `(span, slots, validator outcomes)` → correct/incorrect | Held-out real and synthetic documents |

The extractor trains on **segmented spans, never whole documents**, because section 2.3 proved it cannot handle documents.

### 8.2 Methods worth taking

From **Needle** — proven, and they produced a deployed 45M model:

| Method | Effect |
|---|---|
| Drop the FFN, replace with a fixed Walsh–Hadamard transform | Frees two thirds of non-embedding parameters |
| Reallocate that budget into attention **depth** | Closes the FFN gap to 0.006 nats (0.27%) |
| **QK-normalisation** | The one technique that keeps 48-layer attention-only stacks trainable |
| **Engram memory** — hashed n-gram KV tables | Recovers parametric memory without FFN weights |
| **Depth ladder** — one stack, nested rungs 2→N | One run yields a deployable model family |
| **Overtrain hard** — ~1,200 tokens/param, not Chinchilla's 20 | Cheapest accuracy available on a narrow task |
| QAT from step one | Deployment bits stop being a surprise |
| Grammar-constrained decoding | Verified to terminate cleanly (section 2.7) |

From **`test-model-thing`** — take the ideas, not the code:

| Idea | Verdict |
|---|---|
| Latent-space (JEPA-style) prediction | Interesting, unvalidated |
| Recurrent decay-gated state | Relevant, unproven here |
| **Test-time training** | **Valuable for this problem.** Counterparties, formats and jargon drift; per-deployment adaptation fits payments |
| Byte-level output | Handles any binary input, at a large sequence-length cost |

Its implementation processes **one byte per optimizer step** with a full forward and backward pass per byte — the reason a 4.5M model took twelve hours. Roughly a thousand-fold waste, no tests, no baseline, no metric.

From **NLPBook** — theory, read alongside implementation:

| Chapter | Topic | Applies to |
| --- | --- | --- |
| 6 | Transformers | The architecture being modified |
| 7 | Pre-training | Corpus construction, objectives, scaling |
| 8 | Generative models | Decoding, sampling |
| 9 | Prompting | Conditioning on schema and usage guidelines |
| 10 | Alignment | SFT → preference |
| 11 | Inference | Quantisation, constrained decoding |

Free arXiv companion: **2501.09223**. Pair with `rasbt/LLMs-from-scratch` for code and `smol-course` for alignment.

### 8.3 Quirk mitigations, concretely

| Quirk | Evidence | Mitigation |
| --- | --- | --- |
| **Missing end token → repetition** | Contained for the grammar-constrained call (terminated at 2 chars); unconstrained reasoning rambles | Hard `max_new_tokens`; repetition penalty; n-gram blocking; **structural bound** — emit exactly N values then stop; **stop-token presence check** — if the call does not terminate cleanly, reject the whole extraction rather than trusting a truncated one |
| **Hallucinated fields** | Reasoning invented `bank_name`, `delivery_note`, `receipt_id`, `condition` | Schema-driven allowed-argument list; reject unknown keys |
| **Fabricated values** | `credit_iban: 'CRED'` from "Bank details to follow" | Grounding check (already works, section 2.5) + deterministic validators |
| **Field swaps** | `debtor`/`creditor` reversed on a clean sentence | Positional/role validation; cross-check against span provenance |
| **Non-determinism** | Same input, different outcome | Deterministic backstop; repeated-sampling agreement as a confidence signal |
| **Minor-unit violations** | All three currencies failed | ISO 4217 minor-unit validator (hard reject) |
| **Unconstrained reasoning leaks** | Reasoning contains invented facts | Never emit reasoning into the output document; treat it as a debugging aid only |

### 8.4 Compute

Measured: RTX 2050, compute capability 8.6 (**bf16 works**), 4 GB VRAM, 14 GB RAM, 35 GB disk.

| Model | Tokens | Wall clock |
| --- | --- | --- |
| 10 M | 200 M | ~2.2 h |
| 30 M | 600 M | ~20 h |
| 50 M | 1 B | ~56 h |
| 125 M | 2.5 B | ~347 h |

Local capacity: **5–50 M params to pretrain, up to ~1.5 B to QLoRA fine-tune.** Disk and RAM bind before the GPU — note DocILE alone is large.

- **Local GPU is the development environment.** All debugging here.
- **Cloud is the one big run.** Rented 8×H100 reaches GPT-2 tier in ~1.65 h for ~$48 (~$15 spot).
- **Kaggle** gives ~30 GPU-h/week on T4×2 (16 GB) — build checkpoint/resume first, the session cap is ~12 h.
- **TPUs** (Google TRC, requires a research proposal) need JAX or PyTorch/XLA, so Needle's CUDA path will not run there.

There is a legitimate research question here for a TRC application: *how far does an attention-only extractor go on schema-bound extraction, and does test-time adaptation pay off under document-format drift?* Concrete, falsifiable, and economically motivated.

---

## 9. Evaluation

Since structure is guaranteed by construction, measure what can still be wrong.

| Metric | Definition | Target |
| --- | --- | --- |
| **Slot exact-match F1** | Per-field, on DocILE human-annotated + real documents | Primary quality number |
| **Segmenter recall** | Share of field-bearing spans found | If a span is missed the field is lost |
| **Validator pass rate** | IBAN/BIC/UETR/minor-unit/code-list | 100%, hard reject otherwise |
| **Required-field completeness** | Share of the 174 required paths filled | 100%, else refuse |
| **XSD validity** | Regression canary | 100% by construction |
| **Usage-guideline pass rate** | CBPR+/SEPA layered rules | The production number |
| **Calibration** | Reliability curve | Monotone |
| **Escalation rate** | Share routed to review | The operating cost |
| **Repeat-run agreement** | Same input, N runs | Treat variance as a confidence signal |

The last three decide deployability. A model at 92% with well-calibrated confidence beats one at 95% that is confidently wrong, because only the first can be safely automated. Escalation rate times cost per manual repair **is** the business case.

---

## 10. Phases and gates

### Phase 0 — Foundation (complete)

2,641 XSDs vendored. Introspector extracts element tree, code lists and required paths across `pain.001`, `pacs.008`, `camt.053`. Choice semantics modelled. Needle base verified locally.

Gate passed: introspection works across three message families.

### Phase 1 — Schema compiler (mostly complete)

Serializer proven: shuffled slot order gives byte-identical XML, and output validates against the real XSD.

Remaining: external code sets, ISO 4217 with minor units, IBAN/BIC/UETR/LEI validators, Schematron.

Gate: round-trip fidelity plus validators rejecting deliberately corrupted messages.

### Phase 2 — Synthetic pair generator

Sample → render → pair, using DocILE for layout patterns and the PII distributions for values.

Gate: 100k pairs, every message schema-valid, and a blind pass by someone who did not write the renderer.

### Phase 3 — Rule-based baseline and the economic measurement

No model. Regex, anchors, checksums. **And measure the actual exception economics** — exceptions per day, minutes per exception, loaded hourly cost.

Gate: a real F1 number *and* a real cost-per-exception number. Section 3.3 stands or falls here.

### Phase 4 — Train segmenter and extractor

LoRA first (rank 16, alpha 32, q/k/v/gate/out), then from-scratch once the pipeline is proven. Depth-ladder slice so one run yields deployable rungs.

Gate: beats the Phase 3 baseline on slot F1 **and** on held-out DocILE human annotations the renderer never saw. If it only beats the baseline on synthetic data, the model has learned the renderer.

### Phase 5 — Calibration and gating

Gate on deterministic signals first; add a learned calibration model only if measurement justifies it.

Gate: precision on auto-approved documents clears the business bar, and GDPR Article 22 review obligations are satisfied.

### Phase 6 — Hardening

Adversarial inputs, malformed documents, version skew, repeated-run variance, honest measurement on real messy documents.

Gate: documented delegation path with a measured cost per escalation.

---

## 11. Risks

| Risk | Why it bites | Mitigation |
| --- | --- | --- |
| **Window too small** | Proven failure at 149 tokens | Segmenter is mandatory |
| **No calibration after tuning** | Proven: head returns None | Independent calibration or deterministic gating |
| **Synthetic-to-real gap** | Renderer covers only imagined variation | Hold out DocILE human annotations and real receipts |
| **Template leakage** | Model learns the renderer, not extraction | Many renderers; adversarial novel-phrasing holdout |
| **Confidently wrong** | Worst failure mode in payments | Validators are hard rejects; never auto-approve low agreement |
| **Non-determinism** | Proven: same input, different outcome | Deterministic backstop; repeat-run agreement |
| **Choice-type bugs** | Proven: first XSD validation failed | Now modelled; add a schema-conformance test across three families |
| **Schema-valid but rejected** | CBPR+/SEPA stricter than the base XSD | Layer usage-guideline validation |
| **Version drift** | 2,641 versions; `pain.001` runs v03–v12 | Version as explicit input, never inferred |
| **External code-set churn** | Codes change without a schema release | Pull on a schedule; treat as data |
| **Economic threshold** | Below ~100k msgs/month local is more expensive | Measure volume before committing |

---

## 12. Layout

```text
PROCESS.md                    this document

iso20022-lab/
  pyproject.toml              deps and toolchain config
  probe_failures.py           adversarial probes against this design
  test_architecture.py        the byte-identical + XSD-validity gate
  schemas/                    2,641 vendored ISO 20022 XSDs
  src/iso20022_lab/
    xsd_introspect.py         XSD → element tree, code lists, choice groups
    serialize.py              slots → schema-ordered XML + XSD validation
    # planned
    codesets.py               external code sets, ISO 4217 minor units
    validators.py             IBAN/BIC/UETR/LEI checksums
    segment.py                document → candidate spans
    sample.py                 schema-guided valid message sampling
    render.py                 message → unstructured surfaces
    extract.py                SLM slot extraction
    calibrate.py              independent confidence
    evaluate.py               slot F1, calibration, escalation
```

Reproduce the results:

```bash
source .venv/bin/activate
python iso20022-lab/test_architecture.py    # expect: ARCHITECTURE CLAIM: HOLDS
python iso20022-lab/probe_failures.py       # expect: the failures in section 2
```

---

## 13. References

### Schemas and standards

- ISO 20022 message definitions and machine-readable external code sets — `iso20022.org`
- Vendored XSD set, 2,641 files — `EggBaconAndSpam/iso20022-schemas`
- Parsers, fixtures, candid validation-scope statement — `Query-farm/vgi-iso20022`
- Generator and validator, Apache-2.0 — `sebastienrousseau/pain001`

### Data

- DocILE — `rossumai/docile`, arXiv **2302.05658** (6.7k annotated, 100k synthetic, ~1M unlabeled)
- Invoice Information Extraction: Methods and Performance Evaluation — arXiv **2510.15727**
- HF invoice/receipt corpora as listed in section 7.2
- `gretelai/synthetic_pii_finance_multilingual`

### Theory

- NLPBook — `NiuTrans/NLPBook`, chapters 6–11, and arXiv **2501.09223**
- `rasbt/LLMs-from-scratch` — code companion for every stage

### Small-model training

- Needle — `cactus-compute/needle`; paper arXiv **2607.18363**
- `karpathy/nanochat` — full pipeline behind one depth dial; supersedes nanoGPT, which its own README deprecates
- `Lightning-AI/litgpt` — 20+ models, LoRA/QLoRA, FSDP, TPU/XLA

### Context

- Muon — arXiv **2502.16982**, ~2× compute efficiency at half the optimizer state
- Cramming — arXiv **2212.14034**, single GPU in one day
- TinyStories — arXiv **2305.07759**, why small models work on simple data

---

## 14. MT ↔ MX: the migration corpus

The economic case in section 1 is the CBPR+ migration, so the data that matters is MT and MX for the *same* payments. What exists publicly is tiny, for the same PII reason as everywhere else, but two things are genuinely usable.

### 14.1 The verified golden pair

`Query-farm/vgi-iso20022/data/dual/` ships an MT103 and its pacs.008 for one payment. Both are real, and together they turn the mapping from documentation into observation:

```text
:20:TXN-REF-1                 ->  PmtId/InstrId
:32A:260101EUR1234,56         ->  IntrBkSttlmDt 2026-01-01
                                  IntrBkSttlmAmt Ccy="EUR" 1234.56
:50K:/DE89370400440532013000  ->  DbtrAcct/Id/IBAN
ACME CORP                         Dbtr/Nm
:59:/FR1420041010050500013M02606 -> CdtrAcct/Id/IBAN
WIDGETS SARL                      Cdtr/Nm
:70:INVOICE 998877            ->  RmtInf/Ustrd
:71A:SHA                      ->  ChrgBr SHAR
:72:/EToE/E2E-REF-001         ->  PmtId/EndToEndId
{3:{121:e3bf1c2a-...}}        ->  PmtId/UETR
{1:F01ACMEDEFFAXXX...}        ->  InstgAgt (NOT DbtrAgt)
{2:I103DEUTDEFFXXXXN}         ->  InstdAgt (NOT CdtrAgt)
```

### 14.2 Five traps that survive a naive mapper

Each was found by running the converter against the golden pair, not by reading a specification:

| Trap | Detail | Consequence |
| --- | --- | --- |
| **Charge vocabulary differs** | MT `SHA` → MX `SHAR` | An unrecognised code is a rejection |
| **Decimal separator** | MT `1234,56` → MX `1234.56` | Naive parse yields 123456 |
| **Date format** | MT `260101` → MX `2026-01-01` | Century inference is ambiguous |
| **E2E hides in free text** | `:72:/EToE/<ref>`, not its own tag | A tag-dictionary mapper misses it entirely |
| **Messaging endpoint ≠ agent** | `{1:F01ACMEDEFF...}` is the SWIFT endpoint; `DbtrAgt` is `DEUTDEFF` | Two wrong fields that still look plausible |

The last one is the most dangerous. It produces a message that validates, reconciles against the wrong bank, and is only caught downstream.

### 14.3 Information loss is directional

Round-tripping pacs.008 → MT103 → shared keys recovers 11 of 14 fields. The three that do not come back are **structurally absent from MT103**, not parser failures:

- `DbtrAgt/BICFI` and `CdtrAgt/BICFI` have no MT103 home distinct from the messaging endpoints.
- `PmtId/UETR` survives only because block 3 carries it.

So MX→MT is genuinely lossy. In a dual-running migration that asymmetry must be explicit, or the reconciliation report will show failures that are not defects.

### 14.4 What was built and verified

`iso20022-lab/src/iso20022_lab/mt.py` parses MT103 and translates both directions into the **same key space** the extraction pipeline uses, so one student can serve document extraction and MT→MX migration.

Verified against the corpus in `iso20022-lab/data/`:

| Fixture | Fields recovered | Note |
| --- | --- | --- |
| `payment1.txt` | 14 | the golden pair |
| `valid-envelope.txt` | 12 | correct block 1/2/3/4 framing |
| `cover1.txt` | 11 | MT202 COV cover payment |
| `valid-optional-tags.txt` | 8 | optional tags omitted |
| `missing-59.txt` | 4 | deliberately malformed beneficiary |

Three real bugs were found and fixed by doing this rather than assuming:

1. **Block 4 was never parsed.** A non-greedy `{n:...}` scan stops at block 3's inner brace and then fails its lookahead, silently losing *every tag in the message* — 3 fields recovered instead of 13.
2. **UETR was never extracted.** The block-3 capture starts after `{3:{`, so the field is `121:<uuid>` with no leading brace.
3. **The MT header BIC was malformed.** Block 1 needs a 12-character logical terminal address (BIC8 + terminal + branch); block 2 needs the 8-character form. Appending a branch to an already-11-character BIC produced a 15-character field.

### 14.5 Corpora worth having

| Source | Content | Use |
| --- | --- | --- |
| `Query-farm/vgi-iso20022` | MT104/202/940/942, pacs/camt/pain fixtures **plus `dual/`** | Golden pairs and parsers |
| `Dedmoo/SwiftMt103Parser` | 15 MT fixtures incl. edge cases | Parser testing |
| `sebastienrousseau/bankstatementparser` | MT940/CAMT fixtures, 100% coverage | Statement parsing |
| `qoomon/banking-swift-messages-java` | MT parser/writer | Field semantics |
| `sebastienrousseau/pacs008-loader-mt103` | MT103 → pacs.008 | Reference mapping |
| `moksnow/Mixar` | MT103/202/202COV → pacs.008/009 | Reference mapping |

**Scale is the honest limitation.** 26 files is a seed, not a dataset. The converter plus the XSD sampler is what turns it into volume: sample a valid pacs.008, flatten it, render the MT103, and the pair is correct by construction. That is the same reverse-synthesis move as section 7.3, now applied to the migration task.

---

## 15. The field taxonomy

Sections 2 and 14 reported a chain of build failures that each looked like a model problem and were not. Running the pipeline to completion exposed why: **an ISO 20022 message is built from four classes of field, and conflating them produces failures that misattribute blame to the model.**

| Class | Examples | Supplied by | If you get it wrong |
|---|---|---|---|
| **Extracted** | `Nm`, `IBAN`, `InstdAmt`, `EndToEndId` | The model, from the document | — |
| **Generated** | `MsgId`, `PmtInfId`, `CreDtTm` | The system | A model asked to extract them hallucinates a plausible value that then fails grounding |
| **Derived** | `NbOfTxs`, `CtrlSum` | Computed from the transaction list | Counts disagree with contents |
| **Defaulted** | `PmtMtd=TRF`, `SttlmMtd=INDA` | Fixed by the message type | A model infers from prose a question with one right answer |

Two consequences that only became visible by running it:

**Asking for generated fields guarantees failure.** A document never contains `MsgId`, so every extraction of it is by definition ungrounded. The extraction contract must exclude them — and their absence must then be *filled*, not merely skipped. Excluding without filling moves the failure from the grounding gate to the build gate, where it looks like a schema problem.

**`required` is the wrong filter for choosing extraction targets.** The fields a document actually carries are the *optional* ones and the *choice branches*: `Nm` is optional because a party may be identified by account instead, and `IBAN` is one branch of an `xs:choice`. Selecting on `minOccurs=1` yields `MsgId` and `CreDtTm` and omits every field worth extracting.

One further trap: a required `xs:choice` is a **container**, not a leaf. `ReqdExctnDt` is required, but its satisfiable members are `ReqdExctnDt/Dt` and `ReqdExctnDt/DtTm`. Listing the container in a field priority list matches nothing, because containers are not leaves.

---

## 16. Phase 3: the rule baseline, measured

Section 3.3 said the STP claim needed "exceptions per day, minutes per
exception, loaded cost per hour" and was a hypothesis until those existed. Two
of the three now come from a measurement.

### 16.1 What was built

| Module | Role |
| --- | --- |
| `corpus.py` | Renders real field values into documents, recording exact truth |
| `synth.py` | Validates value synthesis (correct IBAN check digits, minor units) |
| `baseline.py` | The rule engine: label-anchored extraction with pattern fallback |
| `evaluate.py` | Scores an extractor; reports field accuracy and document STP |
| `economics.py` | Prices the exceptions |

The corpus is 201 documents at three difficulty levels, built from the real
MT/MX fixtures plus validated synthetic values. Values are real; **prose is
synthetic**, so every accuracy below is an upper bound.

### 16.2 The measured baseline

| Difficulty | STP | Exception | Field accuracy | Invented/doc |
| --- | --- | --- | --- | --- |
| clean | 85.1% | 14.9% | 98.5% | 0.06 |
| messy | 64.2% | 35.8% | 95.8% | 0.06 |
| hostile | 0.0% | 100.0% | 73.8% | 0.06 |
| **all** | **49.8%** | **50.2%** | **89.3%** | **0.06** |

**The gap between the last two columns is the finding.** Field accuracy is
89.3% and STP is 49.8% — a 39.6-point spread. A payment is not "94% correct":
one wrong field is one exception. Any business case quoting per-field accuracy
is overstating straight-through performance by roughly half.

### 16.3 Rules fail by omission, and models fail by commission

Precision is ~100% on every field. The rules did **not** invent values; they
missed them. Recall is where they collapse: `CdtrAcct_IBAN` 63.6%,
`Amt_InstdAmt_Ccy` 67.7%, `Amt_InstdAmt` 72.1%. Everything else is 92%+.

The cause of the three weak fields is one limitation, not three: **role
disambiguation**. Given `Account: DE89...` the rules know the value is an IBAN
and cannot know whether that account is the payer's or the beneficiary's. That
fact lives in the document's meaning, not its surface form. On a document with
no label, the pattern fallback must guess by position and is wrong about half
the time. That is the honest reason a rules engine plateaus, and it is why
`hostile` scores 0%: at 0%, *every* document has at least one wrong field.

This asymmetry matters more than the rate:

- A rule **miss** produces an exception, which a human completes. Costly, safe.
- A model **hallucination** produces a wrong payment. Cheaper, dangerous.

So a model that beats 49.8% STP on accuracy alone is not automatically better.
Section 2.4 already removed the confidence gate for tuned weights, which means
the hallucination rate above — 0.06/doc for rules — is the number a model must
beat, and it must be near zero.

### 16.4 The number 3.3 was missing

At 100,000 messages/month, 8 minutes per exception, $45/hour loaded:

| STP | Exceptions/month | Hours | Annual cost | FTE |
| --- | --- | --- | --- | --- |
| 85.1% (clean) | 14,925 | 1,990 | $1.07M | 13.1 |
| 64.2% (messy) | 35,821 | 4,776 | $2.58M | 31.4 |
| 49.8% (measured) | 50,249 | 6,700 | **$3.62M** | 44.1 |

**One STP point is worth $72,000/year** at this volume. Sensitivity to the two
soft inputs, at the measured 49.8%:

| min/exception | $25/h | $35/h | $45/h | $65/h | $90/h |
| --- | --- | --- | --- | --- | --- |
| 2 min | 0.5M | 0.7M | 0.9M | 1.3M | 1.8M |
| 8 min | 2.0M | 2.8M | 3.6M | 5.2M | 7.2M |
| 30 min | 7.5M | 10.6M | 13.6M | 19.6M | 27.1M |

### 16.5 Section 3.2 answered a smaller question than it claimed

3.2 concluded that local inference needs ~400k messages/month to beat a hosted
API. Reproducing that arithmetic gives a break-even at **40,000 messages/month**,
and the *direction* of the finding is sound: infrastructure is a volume-threshold
decision.

But it compared **API cost against local-inference cost**, a question worth:

- infrastructure delta at 100k/month: **$56,000/year**

while ignoring the human exceptions *both* options still produce:

- STP delta from 49.8% to 85%: **$2,537,910/year**

**The STP difference is worth 45.3x the infrastructure difference.** At every
volume tested, one STP point exceeds the entire infrastructure spread. The whole
$20k build is repaid by **0.33 of one STP point**.

The corrected statement: 3.2 is an answer about *where inference runs*, and that
question is a rounding error next to *how often the pipeline is right*. The
volume threshold is real; it is simply not where the money is.

### 16.6 What is measured and what is still assumed

Measured: the exception rate, the per-field failure pattern, and the fact that
rules fail by omission rather than commission.

Still assumed: volume, minutes per exception, loaded hourly cost, API price,
build cost. These are inputs a buyer must supply from their own operation and
are exposed as parameters in `economics.py`, swept above so a reader can
substitute their own figures rather than argue with these.

Limitations that bound the result:

- **Synthetic prose.** 201 documents, rendered from variant pools. Real
  correspondence is messier, so both `clean` and `messy` accuracies are upper
  bounds and the true exception rate is likely higher.
- **`hostile` is 0% by construction.** It removes the labels that label-anchored
  rules depend on, so 0% is the expected consequence rather than a discovery.
  It defines the regime where a model is the only option, not a typical one.
- **One baseline, not a population.** A real institution's rules are tuned to
  its own traffic. This is a competent generic engine, so a specific shop may do
  better or worse.
- **No model column yet.** This measures the current process. The comparison
  against a tuned student is Phase 4, and 16.3 says it must report hallucination
  rate alongside STP or it will not mean anything.

## 17. Cost scales with format count, not with volume

Section 16 priced the exceptions at a **fixed** format set and treated the STP
rate as a property of the extraction method. It is not. The STP rate is a
property of **how many document formats you receive**, and that number belongs
to the customer base rather than to the technology.

This section is the answer to "why not just write regexes", and it is a
measurement rather than an argument.

### 17.1 The measured step function

Two rule engines, both competent, both scored on the same 360 documents:

| Extractor | Docs | STP | VERIFY | KEY | Field acc | Invented/doc | XSD pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `rules_v1` | 360 | 33.3% | 0.0% | 66.7% | 47.1% | 0.00 | 33.3% |
| `rules_v2` | 360 | 51.7% | 48.3% | 0.0% | 89.4% | 0.00 | 100.0% |

By difficulty:

| Extractor | clean | messy | hostile |
| --- | --- | --- | --- |
| `rules_v1` | **100.0%** | 0.0% | 0.0% |
| `rules_v2` | 92.5% | 62.5% | **0.0%** |

**`rules_v1` scores 100% on `clean` and 0% on everything else.** Not 80%, not
60% — zero. Both engines were written against the same corpus; the difference is
that `rules_v1` encodes a *specific* template and `rules_v2` attempts every
synonym and a normaliser.

This is the finding: **regex does not fail gradually.** It succeeds completely
inside the format it was written for and fails completely outside it. A rules
engine does not have an accuracy; it has a *coverage*, and coverage is counted in
formats.

The practical consequence is that the cost of a rules approach is not a
percentage of documents. It is **one ruleset per document format**, and format
count is a property of the customer base.

### 17.2 A straw man caught in my own test

The first `rules_v1` scored **0% STP even on `clean`**, and that result was
wrong.

The renderer emits `Amount: EUR 1,250.00` — currency inline with the amount.
The rule looked for a separate `Currency:` label that no document contains. So
the engine failed on a format it was nominally written for, and the 0% measured
a spec mismatch rather than a regex limitation.

Corrected against the format the documents actually use, `rules_v1` scores
**100% on `clean`**. The numbers in 17.1 are the corrected ones.

This matters for how the whole comparison is read: **a baseline that has not
been tuned against the real format is not a baseline, it is a straw man.** Any
"models beat regex" result that does not first show the rules achieving 100%
inside their own format is measuring the author's carelessness, not the
method's ceiling. The honest framing of a rules engine is not "it is 52%
accurate" but "**it is 100% accurate within its format, and there are N
formats.**"

### 17.3 What a human actually does

| Extractor | min/doc | vs. keying everything | Cost / 1k docs |
| --- | --- | --- | --- |
| key from scratch | 8.00 | 100.0% | $6,000 |
| `rules_v1` | 5.33 | 33.3% | $4,000 |
| `rules_v2` | 0.72 | **9.1%** | $544 |

The measured human-effort ratio is where the manual process is actually
replaced. The useful reframe:

> An experienced analyst working a **known** format is not slow. They have
> internalised the template — which is precisely what `rules_v1` encodes. On
> covered formats neither a rules engine nor a model nor a human is the
> bottleneck.

So the 8-minute figure is not the cost of *reading*. It is the cost of
**interpreting an unfamiliar layout** — deciding which account is the
beneficiary's, what an unlabelled amount means, whether a bare reference is
mandated or free-text. That work is per-format, not per-document, and it is the
same work that makes a ruleset expensive to write.

The value of a model is therefore concentrated in the **long tail**: formats no
ruleset covers and no analyst has memorised. `rules_v2` scores **0% on
`hostile`** — that is the tail, measured.

### 17.4 The format-concentration model

If cost scales with formats, the comparison has a crossover rather than a
verdict. `format_economics.py` computes it.

Assumptions, all exposed as parameters: rules cost **2 engineer-days per format**
plus 0.5 days/year maintenance at $600/loaded-day; the model is **25 build + 10
integration days** fixed, plus 6 days/year, with **no per-format term**. Volume
100k/month; KEY 8 min, VERIFY 1.5 min, $45/hour. Volume is spread over formats
with a Zipf exponent, since inbound traffic is dominated by a few templates with
a long tail.

| Formats | Volume rules cover | Rules year 1 | Model year 1 | Cheaper |
| --- | --- | --- | --- | --- |
| 3 | 100.0% | $3,600 | $969,375 | **rules** |
| 5 | 80.3% | $1,424,978 | $969,375 | model |
| 10 | 62.6% | $2,705,294 | $969,375 | model |
| 25 | 48.0% | $3,770,843 | $969,375 | model |
| 250 | 30.1% | $5,336,305 | $969,375 | model |

**The crossover is 4–6 formats**, depending on how concentrated the traffic is
(4 at exponent 0.6, 5 at 1.0, 6 at 1.5).

The sensitivity check is what makes this robust rather than a sales figure:

| Assumed model STP | Crossover |
| --- | --- |
| 60% | 7 formats |
| 70% | 6 formats |
| 80% | 5 formats |
| 90% | 4 formats |

**Even at 60% STP — below what `rules_v1` achieves on its own format — the model
is cheaper once the format count is large.** The argument does not require the
model to be more accurate than regex inside a format. It requires only that
regex needs one ruleset per format and a model does not.

### 17.5 The term that dominates: onboarding

Not in the cost table, because it is lead time rather than run rate:

| | Rules | Model |
| --- | --- | --- |
| Engineering per new format | 2 days | 0 |
| Cost | $1,200 | $0 |
| Lead time | days to deploy | none |

Every new customer with a new template costs the rules approach $1,200 and a
deployment cycle, and costs the model approach nothing. **That is time-to-revenue
on every customer won**, and on a per-customer basis it exceeds the entire
annual maintenance delta in the table above.

This is also the sharpest statement of when *not* to build the model. A shop
with three stable formats and no acquisition pipeline should write the rules:
they are exact, deterministic, auditable, and cost $3,600 against $969,375.

### 17.6 What this changes about section 16

16.4 computed **one STP point is worth $72,000/year** at 100k messages/month,
and 16.5 showed the STP delta dominates the entire infrastructure decision by
45x.

Those hold. What changes is where the STP rate comes from. It is not a fixed
property of the extraction method to be improved by a few points — it is a
function of format coverage, and format coverage is a function of how many
different senders you accept. **The dominant input to the business case is the
customer count, not the model.**

So the decision procedure is:

1. Count the document formats arriving per month.
2. If it is under roughly three and stable, write rules and stop.
3. If it is four or more and growing, the rules approach is a per-customer
   engineering cost and the model is the cheaper instrument — at any plausible
   accuracy.
4. Price the STP change with 16.4, and the format coverage with 17.4. Do not
   conflate them; they are separate terms.

### 17.7 Measured and assumed

**Measured:** the rules' step function across difficulty levels, the
role-disambiguation failure pattern from 16.3, the human-effort ratio (5.33 and
0.72 min/doc), and the reproduction of `rules_v1` at 100% on its own format
after the spec correction in 17.2.

**Assumed:** engineer-days per format, loaded day rate, maintenance rates, build
and integration days, volume, minutes per exception, hourly cost, and the Zipf
concentration. All are parameters in `format_economics.py` and are swept so a
reader can substitute their own operation's numbers rather than argue with
these.

### 17.8 Limitations

- **The corpus is equal thirds by construction.** 120 clean / 120 messy / 120
  hostile does not reflect any real inbound distribution. Real concentration is
  a business input that cannot be measured from here, which is exactly why the
  crossover is swept over the Zipf exponent instead of asserted at a point.
- **`format count` is proxied by the Zipf exponent**, not observed. A shop whose
  formats are uniformly distributed lands at the optimistic end.
- **Two rule engines, not a population.** A real institution's rules are tuned
  to its own traffic and may beat both; the step-function shape should survive
  that, the specific rates may not.
- **The model column is still modelled, not measured.** 17.4 assumes 85% STP at
  the base case. The sensitivity sweep exists because that number is not yet
  evidence, and Phase 4 must replace it.
- **Onboarding cost is not fully captured.** A ruleset also needs regression
  tests and monitoring per format, which raises the per-format term; the model's
  equivalent cost is a retraining cycle, which is not per-format. The direction
  is unaffected.

## 18. Inbound capture: five channels, and OCR is needed by almost none of them

Section 17 answers *whether* to build the extractor. This section is about what
happens **before** it — getting unstructured inbound into text — and it exists
because the naive answer is expensive and wrong.

The naive design is "OCR everything, then extract". It is wrong for three
reasons. Most inbound is already text. Recognition is lossy, so running it on
good text actively degrades it. And it is the slowest and most failure-prone
stage in the pipeline, so paying for it unnecessarily costs both money and
accuracy.

### 18.1 The channels

| Channel | What arrives | Recognition needed? |
| --- | --- | --- |
| Email body | plain or HTML text | **never** — no pixels hold the data |
| EDI (MT101, pain.001) | an actual message | **never** — parse it |
| PDF / portal upload | text layer present | only if the layer fails its checks |
| PDF / portal upload | scan, no text layer | **yes** |
| Fax | pixels, definitionally | **yes** |

Two rules follow, and both are in the router as tests:

**A structured message is never recognised.** Running OCR over a `pain.001` that
already parsed converts correct data into a guess. That is a pure loss.

**A text layer is probed, not trusted.** A PDF's embedded text layer is
frequently *present but wrong* — ligature mapping failures, CID font
substitutions, broken reading order. "It has a text layer" is not evidence that
the text layer is right.

### 18.2 The probe: output validators used as input quality control

Detecting a corrupt text layer normally means running OCR and diffing, which
defeats the purpose. Instead the capture stage reuses the **deterministic
validators already used to gate the output** — IBAN mod-97, BIC country codes —
as an input-quality probe.

The reasoning is that an IBAN in the source document has correct check digits *by
definition*. So any text layer producing IBANs that fail mod-97 has corrupted
them, and there is no need to know the right value to know this one is wrong.
The validators earn their keep twice, and the probe is deterministic, auditable
and free.

Measured over 43 genuinely corrupting transformations and 200 clean documents:

| Metric | Result |
| --- | --- |
| Corruption detection (IBAN corruptions) | **43 / 43 = 100%** |
| False rejects (clean documents) | **0 / 200 = 0%** |
| Formatting variants correctly accepted | **12 / 12 = 100%** |

Per corruption class:

| Corruption | Caught | | Corruption | Caught |
| --- | --- | --- | --- | --- |
| `0`→`O` | 6/6 | | adjacent transpose | 6/6 |
| `1`→`l` | 6/6 | | dropped space | 6/6 |
| `5`→`S` | 4/4 | | digit flip | 6/6 |
| `8`→`B` | 4/4 | | `2`→`Z` | 5/5 |

The counts differ per class because a substitution does nothing to an IBAN that
does not contain that digit. Those cases are excluded rather than counted as
successes; counting them is one of the measurement bugs in 18.3.

### 18.3 Four bugs that only running the tests found

This section is worth reading for these, because every one of them was invisible
to reasoning and would have shipped.

**(a) The measurement itself was wrong.** The first version of the test grouped
corrupting and non-corrupting transformations into one list and reported a
single "detection rate" of **81.7%**. Every miss was the measurement's fault:

- `insert-space` and `lowercase` produce a value that **normalises back to the
  correct IBAN**. Accepting them is correct, not a miss.
- `five->S` does nothing to an IBAN containing no `5`. A no-op cannot be caught,
  because nothing changed.
- `lowercase` was scored as *caught* when the probe had merely found no IBANs at
  all. That is a different outcome being counted as a detection.

The fix was to separate CORRUPTING from FORMATTING and to drop no-ops. The
correct number was never 81.7%; it was 100%, and the 81.7% measured the test.

**(b) `\s?` matched a newline, rejecting 77% of clean documents.** The IBAN
pattern used `\s?` for separators. `\s` matches `\n`, so on text laid out one
field per line the pattern ran **across the line break** and consumed the next
label — `DE89...3000` followed by `\nAccount` matched as one 22-character token,
failed its checksum, and condemned the whole document. Measured effect: a **0.77
false-reject rate**, sending most clean PDFs to OCR for nothing. The fix is a
literal space: spacing inside an IBAN is a space, and a newline means the field
ended. This is the single highest-value bug caught here.

**(c) The BIC check was shape-only, and English words passed it.** `bic_valid`
matches `[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?` and nothing else. Measured:
`attached`→`CH`, `beneficiary`→`FI`, `instruction`→`RU`, `transferred`→`SF`,
`received`→`IV`. Three of those are real countries. The dangerous direction is a
false **valid**, which *raises* the trust score and manufactures confidence in a
document where nothing was verified.

**(d) Adding country validation was necessary and not sufficient.** ISO 9362 puts
a real country code at positions 5-6, so validating it is correct — and it does
not reject `attached`, because `CH` genuinely is Switzerland. A BIC-shaped
English word with a valid country code is **structurally indistinguishable from a
BIC by pattern matching**. The fix was to require context: a BIC-shaped token
counts as evidence only when a BIC label precedes it on the same line. A token
behind a label is evidence; a token mid-sentence is a coincidence.

Two routing claims were also wrong when stated loosely and had to be rewritten
twice. The true claim is narrower than "only two of six channels need OCR": it is
that **email bodies and EDI never need recognition**, and everything else needs
it exactly when no text layer is present. The first version of the claim was
about channels when the real variable is whether pixels are the only
representation available.

### 18.4 How this is better than regex, concretely

Extending section 17's argument to the capture stage:

- **Regex cannot read a fax at all.** Not "performs worse" — there is no input.
  For fax and scans, recognition is not an optimisation, it is the only path.
- **A text layer is not a guarantee, and regex cannot tell.** A rule engine reads
  whatever characters the layer produced, corruption included. The probe
  validates the layer *before* extraction, using checksums rather than pattern
  shape, so a corrupt layer is caught rather than faithfully parsed into a wrong
  payment.
- **Regex needs one pattern per layout; the probe needs none.** The probe works
  on any layout because it validates values, not positions.

And the honest boundary: **for a text-layer PDF, a rule engine reading the same
text is not worse.** The gain is not that extraction works better; it is that OCR
is skipped when the text is already good, and triggered when it is not. That is a
routing improvement, and it is the part that is measured.

### 18.5 PaddleOCR: the ingest engine, and what is not tested

The integration point is `PaddleOcr`, written against the PaddleOCR 3.x API:
PP-OCRv6 for detection and recognition, PP-StructureV3 when layout matters
(multi-column remittances, tables of invoices).

The reason to use a sub-100M specialist rather than a document VLM is the same
bet as sections 4 and 17: PaddleOCR's own technical report argues specialists of
this size rival billion-parameter VLMs at OCR, and at 100k documents/month a
local sub-100MB model on CPU is a far better operating point than a hosted
multimodal call — and it keeps payment data on-premises.

**Status update: this is no longer untested.** The ONNX weights now run through
`ocr.py` on `onnxruntime`, with no PaddlePaddle dependency, and section 18.8
records what the first real end-to-end run produced. The paragraph below is kept
because it is still true of the *PaddleOCR 3.x wrapper* specifically, which
remains unexercised.

**What was not tested at the time of writing:**

- PaddleOCR is **not installed in this project** and there are **no rasterised
  pages** in the corpus, so `PaddleOcr` has never executed. It is the integration
  point, shaped to the documented API.
- The OCR path is exercised through a **stand-in backend**, which proves the
  wiring — a rejected text layer reaches the backend and its text is returned —
  and proves nothing about recognition accuracy.
- Orientation classification, un-warping, and PP-StructureV3 layout extraction
  are all **unverified here**. They must be tested against real scans before
  anyone relies on them.
- Handwriting remains a known weak point for this class of model and is not
  addressed by anything in this section.

### 18.6 Measured and assumed

**Measured:** the routing decisions across channels and text-layer states; the
probe's detection rate on 43 systematically generated IBAN corruptions; its
false-reject rate on 200 clean documents; its acceptance of 12 formatting
variants; and, as a regression guard, that the existing suite still passes.

**Assumed:** that real text-layer corruption resembles the corruptions generated
here. The transforms are drawn from documented failure modes — glyph substitution
and digit transposition — but they are *generated*, not sampled from real broken
PDFs, because no real broken PDFs are available in this project.

### 18.7 Limitations

- **Synthetic and injected corruption.** The probe's 100% detection is against
  corruptions this project produced. Real text layers fail in messier ways that
  may not produce checksum-invalid values at all — dropped fields, reordered
  columns, truncated lines. Those are undetectable by a checksum probe and would
  pass as "trustworthy" if the surviving IBANs are intact.
- **The probe is blind without checksummed fields.** A document with no IBAN and
  no BIC gets no verdict, and the module returns `trustworthy=False` on that
  basis. That is the safe default — it routes to OCR — but it means documents
  without account numbers always cost OCR. The probe cannot distinguish "clean
  but no IBANs" from "corrupt".
- **Detection is per-document, not per-field.** One bad IBAN rejects the whole
  layer. Given 0% false rejects on clean documents this is currently free, but
  the threshold is a parameter because that trade is a business decision.
- **Checksums bound the probe's reach.** It catches digit and transposition
  errors, which is what report section 16.3 shows is the dangerous class. It does
  not catch a correctly-read value assigned to the wrong role, which is 16.3's
  dominant failure mode and remains the extractor's problem.
- **The OCR engine is untested here**, as stated in 18.5. Nothing in the measured
  results above is evidence about recognition quality.

### 18.8 PP-OCRv6 ran for real, and the XSD gate did not catch what it produced

Section 18.5 said the OCR path was an untested integration point. It has now been
run against the actual PP-OCRv6 ONNX weights (tiny detection 1.8 MB + tiny
recognition 4.5 MB, via `onnxruntime`, no PaddlePaddle). The first end-to-end
result is the most useful thing in this document, and it is not a success story.

**The pipeline works mechanically.** A 32px rendered payment instruction was
detected and recognised in 1.1 seconds at 0.954 mean confidence, fed to the rule
extractor, validated, and built into a message that **passed the XSD schema**.

**Every field in that message is wrong.**

| Field | Produced | Actually is |
| --- | --- | --- |
| `CdtrAcct_IBAN` | `DE89370400440532013000` | the **debtor's** account |
| `DbtrAcct_IBAN` | `GB29NWBK60161331926819` | the **creditor's** account |
| `Amt_InstdAmt` | `29.00` | should be `1250.00` |
| `PmtId_EndToEndId` | `DE89370400440532013000` | an IBAN, not a reference |
| `CdtrAgt_BICFI` | `INSTRUCTION` | an English word |
| `ReqdExctnDt_Dt` | `2024-03-15` | correct |
| `Cdtr_Nm` | `Beta Ltd` | correct |
| `Dbtr_Nm` | `Acme GmbH` | correct |

`XSD valid: True`. The schema validated a payment going to the wrong account for
the wrong amount, out of an account that is not the payer's.

**Why the gate passed it.** Every individual value is well-formed. Both IBANs are
genuine IBANs with correct check digits, so `iban_valid` is satisfied by either
one. `29.00` is a well-formed amount in a 2-decimal currency. The XSD checks
structure and datatypes; it has no opinion about whether the value in a slot came
from the right place in the document. The validators check *shape*, and the shape
was correct.

**What this means for the three checks in this document:**

- Section 16.3 said "an extraction that cannot become a valid payment has not
  solved the workflow". That is true, and this run shows the **converse is
  false**: an extraction that *does* become a valid payment may still have solved
  nothing. XSD validity is necessary and nowhere near sufficient.
- Section 16.3 also named role disambiguation as the dominant failure mode. This
  is that failure, reproduced on a real OCR read rather than a synthetic corpus,
  with the roles swapped.
- The acceptance gate's C7 (XSD round-trip) is therefore a **necessary but
  insufficient** acceptance check. It must never be reported alone, and a model
  that passes C7 at 90% has proved almost nothing. C5 (field accuracy against
  ground truth) is what carries the weight, and C4 does not help here either,
  because every wrong value is traceable to the document -- it is grounded, and
  it is in the wrong slot.

**Two OCR-specific bugs were found and fixed by running it**, both of which looked
like model weakness and were not:

1. **The detector downsampled and the code treated map coordinates as image
   coordinates.** Boxes came back 7px tall for 28px text, recognition received a
   thin horizontal slice of each line, and produced `Dabtor` for `Debtor` and
   `FUR` for `EUR`. Fixed by scaling boxes by the map-to-image ratio before any
   filtering or expansion.

2. **`unclip_ratio` was too small, and its default is the published PP-OCR
   value.** Measured on a 32px render:

   | ratio | crop height | exact | normalised |
   | --- | --- | --- | --- |
   | 1.6 (PP-OCR default) | 18–19px | 0/8 | 3/8 |
   | 2.0 | 24–25px | 0/8 | 3/8 |
   | 2.5 | 30–31px | 3/8 | 7/8 |
   | 3.0 | 36–39px | 2/8 | 8/8 |

   At the shipped default, crops were half the height the text actually needed.
   2.5 is now the default: 3.0 scored better on this fixture but expands boxes to
   36–39px against a 54px line pitch, and a merged line corrupts two fields at
   once, which is worse than a misread glyph.

**The dictionary is also load-bearing and a mismatch is silent.** Recognition
emits one class per dictionary entry plus a CTC blank plus an optional space:
`tiny_rec` has 6906 classes against `ppocrv6_tiny_dict.txt` (6904 entries), and
`medium_rec` has 18710 against `ppocrv6_dict.txt` (18708). The v5 dictionary
(18383) and v1 (6623) both fit neither. A wrong dictionary does not error -- it
decodes to confident nonsense, because every index still maps to *a* character.
`load_dictionary` now refuses to run on a count mismatch.

**What is still not tested.** Single clean fixture, 8 lines, one font, one size,
no rotation, no skew, no noise, no multi-column layout, no handwriting. The
medium models are unexercised. And the finding above is a *reason to distrust*
the XSD gate rather than evidence the whole path works: OCR read the page well
enough that a human reading the same text would have gotten the roles right, and
the pipeline still produced a wrong payment that validated.

### 18.9 C8: the check that catches what every other check passed

Section 18.8 recorded an extraction that was XSD-valid, used canonical keys, had
no duplicate keys, and whose every value was traceable to the document -- and
which sent the payment to the wrong account for the wrong amount. C1 through C4
and C7 all passed it. C5 would have caught it, but **C5 is accuracy against ground
truth, and ground truth does not exist at inference time.**

So the gate needed a check that needs no ground truth. C8 compares the extraction
against **the source document's own role labels**, which is the only authority
present at run time.

| Sub-check | What it asserts | What it caught |
| --- | --- | --- |
| 8a | debtor and creditor accounts differ | -- |
| 8b | a role's value is not the value the source labels the other way | both IBANs |
| 8c | the amount appears in the source in an amount context | `29.00` |
| 8d | a BIC value is anchored to a BIC label in the source | `INSTRUCTION` |

Measured against the real failing extraction and against a correct one:

```text
BAD  (the actual PP-OCRv6 run)  -> 4 findings, all correct
GOOD (roles correct, no BIC)    -> 0 findings
GOOD (roles correct, anchored BIC) -> 0 findings
```

**Two of these sub-checks were wrong in their first form**, and both were fixed by
running them rather than by reading them:

- **8c tested the wrong mechanism.** It checked whether the amount was a substring
  of an account identifier. It never fired, and the premise was false: `29.00` is
  not carved out of `GB29NWBK...`, it is **spliced** -- `29` from the account's
  country-check digits and `.00` from the real amount. A splice is not a
  substring, which is exactly why it is hard to see. The check now tests whether
  the amount appears in the source in an amount context, which the splice fails.

- **8d was fooled by the same trap as the capture probe.** It assumed a long
  alphabetic value would not be BIC-shaped. `INSTRUCTION` is 11 characters,
  matches `[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}[A-Z0-9]{3}` exactly, and its positions 5-6
  are `RU` -- Russia, a real ISO country. It passes `bic_valid` *and*
  `bic_country_valid`. The same conclusion as section 18.3, reached independently
  at the other end of the pipeline: **shape plus a real country code is not
  evidence.** 8d now requires context anchoring.

**Thresholds.** 8a-8d are set at 1.00 rather than a tolerance, unlike C5 and
C5b. A misread field degrades accuracy; a role inversion sends money to the wrong
place, and there is no downstream control that catches it -- the XSD schema passed
the exact extraction this check exists to reject. For this failure mode, a 98%
tolerance is a 2% rate of wrong recipients.

**What C8 still cannot do.** It detects values in the wrong *place*, never values
that are wrong *in place*. A misread IBAN whose check digits happen to still
validate, in the correct role slot, passes C8 and C7 and every structural check.
That residual is why C5 exists and why the acceptance gate cannot be run without
held-out documents that carry ground truth.

### 18.10 The model finished training, and it does not work

Training completed: stage 1, 1071 steps, 3 epochs, final loss 1.1244, and a
**token accuracy of 82.14% on the JSON target**. Stage 2 then reported something
flatly contradictory:

```text
samples                  400
exact_match_rate         0.0000
mean_field_recall        0.0000
```

Zero across 400 held-out documents. The acceptance gate was run to resolve the
contradiction, and the resolution is that **stage 2 was right and stage 1's number
was misleading**:

```text
[FAIL] C1 valid JSON object    0.0833   1/12 parsed
[FAIL] C2 canonical keys only  0.0000   0/12
[FAIL] C3 no duplicate keys    0.1667   2/12
[FAIL] C4 values grounded      0.0000   0/12
[FAIL] C8 role consistency     0.0000   0/1
[FAIL] C5 field accuracy       0.0083   1/120 fields correct
[FAIL] C5b document STP        0.0000   0/12 documents fully correct
[PASS] C6 deterministic        1.0000   3/3
[FAIL] C7 XSD round-trip       0.0000   0/36 built a schema-valid message

VERDICT: FAIL -- do not publish
```

**One field out of one hundred and twenty.** A representative generation:

```text
{"Amt_InstdAmt":"198934516","Amt_Ccy":"EUR","CdtrAcct_IBAN":"DE9695792208795204",
 "CdtrAgt_BICFI":"DEUTDEFF","Cdtr_Nm":"Orchid Pharma BV",
 "Dbtr_Nm":"Orchid Pharma BV","Dbtr_Nm":"Orchid Pharma BV",
 "PmtId_EndToEndId":"REF-6641208654111249213411341..."}<|end|>82"}<|end|>82"}...
```

Duplicated keys, an invented `Amt_Ccy`, a name in a BIC field, values appearing
nowhere in the document (`DE9695792208795204`, `198934516`), and a repetition loop
that never emits a terminator, so the output is not JSON and `parse_generation`
returns `{}`.

**Why 82% token accuracy and 0% output are not a contradiction.** Stage 1 is
teacher-forced next-token prediction: at every step the model is handed the
*correct* prefix and asked for the next token. It scores 82% on that. At
generation time it receives its own output, so a single wrong token is fed back
and everything after it is conditioned on an error. This is exposure bias, and it
is a known property of the training objective rather than a bug in it -- but the
magnitude here is total, not marginal.

**The lesson is about the metric, not the model.** Token accuracy on a
teacher-forced target is the number this project has been quoting since section
8, and it was **never a valid predictor of whether the artifact works**. A metric
that reads 82% on a model that extracts 1 field in 120 is worse than no metric,
because it looks like progress. The only number that should have been reported at
each checkpoint is the one stage 2 computes: generate, parse, score against truth.
Those evaluations existed and were logged (71.03%, 73.11%, 76.06%, 79.25%,
82.14%) and they tracked a quantity that does not matter.

**What this does not tell us.** It does not say the architecture is wrong. The
model is 53M parameters trained for 1071 steps on 4000 synthetic documents on a
4GB laptop GPU; the training configuration in section 8 was written for a much
longer schedule, and 3 epochs at this scale is an under-training result, not a
capacity result. What it does say is that **the current process cannot tell the
difference between under-training and a broken approach**, because it was
measuring the wrong thing. Fixing the measurement comes before spending more
compute.

**The gate earned its keep.** It was built specifically because "does it work" had
been an assumption, and it returned a decisive, diagnosable FAIL with the failing
documents attached. Nothing was published. C6 passing at 1.0 is correct and
expected: greedy decoding is deterministic even when the output is wrong, because
it tests a property rather than a quality.

**One C8 false positive worth recording.** `CdtrAgt_BICFI='DEUTDEFF'` was flagged
as unanchored, but the source document reads `Beneficiary BIC: DEUTDEFF698`. The
model produced a truncated-but-recognisable BIC rather than an invented one. C8's
8d requires an exact match against the labelled value and therefore reports a
partial read as a provenance failure. That is the wrong diagnosis for this case,
and the check should accept a prefix match before the next run.

### 18.12 The capture layer, and why OCR is not the default

Section 18 described five inbound channels and asserted that OCR is needed by two
of them. That assertion is now measured, and the routing table has code behind it
where before it had only prose.

**The measured recognition tax.** Same 135 documents, same rule engine, two
capture routes:

```text
  rules on exact text        fields  88.96%  STP  52.59%  1201/1350
  rules on recognised text   fields  14.67%  STP   0.00%   198/1350

  field accuracy  +74.30%      document STP  +52.59%      wall clock  4296x slower
```

Reproduce: `python -m iso20022_lab.measure_capture --docs 45 --seed 7 --ocr tiny`

The correct baseline is not "regex versus OCR". It is **rules on exact text**
(what a text layer gives) versus **rules on recognised text** (what a fax gives).
The gap is the recognition tax, and it is 74 points of field accuracy and the
entire STP rate. For a document that already carries exact characters,
recognition can only subtract information.

**Damage attributed by cause**, because 74% is not actionable on its own:

```text
  role      73  a real value from this document put in the wrong field
  missing  877  nothing extracted; incomplete, so review sees it
  caught   193  characters changed and a validator rejects it
  silent     9  characters changed and nothing rejects it

  visible to existing checks : 1070
  INVISIBLE to all of them   :   82
```

Only 82 of 1070 errors are invisible. That is the number worth engineering
against, and it is 7.7% of the damage rather than 74%.

Two attribution errors were made and corrected while producing this table, and
both are worth recording because each would have sent the work to the wrong
component:

- **Role errors were counted as recognition damage.** The first output showed
  `ES684...` becoming `ES091...` and `ES091...` becoming `ES684...` — the same two
  valid account numbers, swapped. Recognition cannot swap them; only a role
  collapse can. Recognition *causes* the collapse indirectly, by damaging a label
  (`Beneficiary IBAN` came back as `BaneficiarYIBAN`), which costs the extractor
  its anchor and forces positional guessing. The distinction matters because the
  fix differs: harden label matching, not the recogniser.
- **Missing values were counted as silent.** A missing field is a *visible*
  failure — the document is incomplete and review sees it. Only corruption that
  passes validation is silent.

**The finding that matters: BIC has no checksum.** All 9 silent errors are
`CdtrAgt_BICFI`, and **8 of 9 pass both `bic_valid` and `bic_country_valid`**:

```text
  BNPAFRPP704  ->  BNPAERPP704      FR (France)      -> ER (Eritrea)
  ABNANL2A836  ->  ARNANI2A836      NL (Netherlands) -> NI (Nicaragua)
  DEUTDEFF349  ->  DEUTDEEE349
  UNCRITMM     ->  UNCDITMM
```

IBAN, under the same recognition: **236 damaged, 0 silent**. mod-97 catches every
single-character change. A BIC has no check digit at all, so a character
substitution yields another well-formed BIC — and frequently one for a different
real country.

**This corrects a claim the code was making.** `validators.bic_country_valid`
documented itself as rejecting prose: *"`attached`, `beneficiary` and
`instruction` all pass `bic_valid`; none passes this."* **All three pass.** Their
positions 5-6 are `CH` (Switzerland), `FI` (Finland) and `RU` (Russia) — all real
countries. A test written for the recipe book caught it.

The correction strengthens the argument. If prose passes *and* a corrupted BIC
passes, then **shape plus a real country code is not evidence about a BIC at any
point in the pipeline** — not on the way in, not on the way out. What actually
rejects prose is C8's context anchoring (§18.9). The country check discriminates
*between prose and a value*; it cannot discriminate *between a value and a
misread value*. Two threat models, and one check cannot serve both.

This is the same conclusion §18.3 reached at the output end — where `instruction`
matched the BIC pattern and `RU` is a real country — arrived at here from input
pixels, and §18.11 reproduces it on a clean render.

**What was built.**

| Module | Provides |
| --- | --- |
| `documents.py` | bytes → `capture()` inputs: sniffing, PDF, email, dispatch |
| `measure_capture.py` | the measurement above, with per-field damage attribution |
| `fields.py` | the canonical field space and its parsers, importing nothing |
| `CAPTURE.md` | the design document |
| `RECIPES.md` | the recipe book, every recipe paired with a test |

Three bugs were found by running the tests rather than reasoning about them:

- **A raw email was classified as `text/plain`.** A message is ASCII, so the text
  fallback claimed it, and the *entire MIME envelope — base64 attachment payload
  included — was handed downstream as the document's text*. That is a silent
  wrong-value path, not a parse error: base64 blobs in the "document text" that
  extraction would scan for field values. Fixed by detecting an RFC 822 header
  block before the text fallback.
- **HTML was classified as `application/edi`.** The EDI marker regex accepted a
  bare `<` for XML forms like pain.001, and `<` also opens every HTML email.
- **`extract_image` hardcoded `media_type="image"`**, discarding the sniffed
  format and the distinction between a scan and a photograph.

And one structural fix: importing the scoring primitives into `train.py` from
`acceptance.py` created

```text
acceptance -> workflow -> model.evaluate_model -> model.train -> acceptance
```

which pyright reported as `Cycle detected in import chain`. A deferred import
would have hidden the report while leaving the layering wrong. The real problem
was that **pure functions over strings lived in the module that knows about
trained models, XSD round-trips and human workflow.** They now live in `fields.py`,
which imports nothing from the package, so the cycle cannot recur — and which also
removes the three-parsers drift that §18.10 recorded.

**End-to-end, every channel, one complete document:**

| Channel | Route | OCR | Fields | Schema-valid message |
| --- | --- | --- | --- | --- |
| Plain text body | `text` | no | 10 | **yes** |
| PDF with text layer | `probe` | no | 10 | **yes** |
| Email + PDF attachment | `probe` | no | 10 | **yes** |
| PDF scanned | `ocr` (rasterise first) | yes, conf 0.78 | 3 | no |
| Fax / raw image | `ocr` | yes, conf 0.93 | 5 | no |

The free paths are perfect and the OCR paths lose the message, exactly as the
measured tax predicts. **"PDF scanned → OCR" now has a code path**: before
`documents.py` the routing table promised it and nothing turned a PDF into pixels.

Note that an incomplete extraction failing to build a valid message is the system
working, not failing. An early version of this test used a six-line fixture, and
the XSD correctly rejected the resulting payment for missing `ReqdExctnDt` — the
builder returns `ok=True` on all 9 complete documents and refuses partial ones.

### 18.13 Closing the BIC gap, and a measurement that does not describe the artifact

**Part one: the residual gap is closed.** Section 18.12 ended with an open item --
all 9 silently-wrong fields were the agent BIC, and 8 of 9 passed shape *and*
country checks. The fix is `routing.py`, and it is not a better recogniser.

The asymmetry is the tool. `CdtrAcct_IBAN` is checksum-protected so its country is
trustworthy; `CdtrAgt_BICFI` is not; they describe the same party. So the
trustworthy field tests the untrustworthy one:

```text
CdtrAcct_IBAN = FR7630...   (mod-97 verified)
CdtrAgt_BICFI = BNPAERPP704 (Eritrea)
-> the account is French, the bank is claimed to be Eritrean: reject
```

That catches 8 of the 9. The ninth, `DEUTDEFF349` read as `DEUTDEEE349`, keeps its
country -- so no country check can see it, and `routing.py` says so rather than
implying coverage it lacks:

```text
[ok  ] bic_country_matches_iban: creditor BIC country DE agrees with the account country
[??  ] bic_matches_directory: creditor IBAN is not in the routing directory (none),
        so the bank code cannot be confirmed -- only the country was checkable
```

`UNVERIFIABLE` is deliberately never folded into `PASSED`. With an
`ObservedDirectory` holding one settled correspondence, the ninth case is caught:
*same country, different institution*. Wired into the gate as C9 at 1.00. 18
tests, every failure case taken from the measured run rather than invented.

**One bug found by writing those tests.** `cross_check` resolved its directory with
`directory or NoDirectory()`. `ObservedDirectory` defines `__len__`, so an *empty*
one is falsy -- and a configured-but-unpopulated directory was silently replaced by
"no directory configured". Nothing failed loudly; the check still said
unverifiable, just for the wrong reason. `is None` is the fix.

**Part two: the training metric does not describe the saved artifact.**

The 3-epoch run reached 82.14% token accuracy and extracted 1 field in 120, which
is why the generation-based metric was built. Fourteen epochs were then run, and
the log reported near-perfection from step 1400 onward:

```text
EVAL step 2400: loss 0.0129 json 100% term 100% keys 100% nodup 100% recall 100.0% stp 100.0%
```

Measured independently, on the same construction the trainer uses, that same
checkpoint gives:

```text
step2000   json  0.0%  recall  0.0%  stp  0.0%
step2400   json  0.0%  recall  0.0%  stp  0.0%
final      json 50.0%  recall  5.0%  stp  0.0%     (the earlier 3-epoch run)
```

**The saved checkpoints are measurably worse than the earlier run, while the log
claims perfection.** A minimal in-process reproduction -- train 30 steps, then
compare the live model, the saved checkpoint, and the metric, all in one process --
showed the three **agreeing at 0%**. So the metric, the save, and the load are each
sound in isolation, and the discrepancy is specific to the full-size run.

Narrowed but not closed. What was ruled out, each by execution rather than
reasoning:

- **The export.** All 214 tensors load identically; raw checkpoint and export
  produce the same output.
- **Weight tying.** `lm_head.weight` is genuinely the same storage as
  `embed_tokens.weight`; save drops it (correct) and reload re-ties it; the
  round-trip is byte-identical.
- **The config round-trip.** Only cosmetic differences (`tuple` -> `list`,
  `transformers_version`).
- **The 213-vs-214 tensor count.** `named_parameters()` deduplicates the tied
  pair; that is expected, not corruption.
- **Evaluation order.** `evaluate()` leaves the model in train mode, but
  `generation_metrics` sets eval and the order makes no difference.
- **The parser.** `parse_strict` is the single implementation now, and a
  `loss=nan` seen once did not reproduce.

**One real bug found along the way, and it is relevant.** `split_examples` uses
`torch.randperm(len(items))`, so the validation set depends on **corpus size**,
not only on the seed. `build_examples(4000)` returns 12,000 documents (4,000 seeds
x 3 difficulty levels) and `build_examples(200)` returns 600; their validation
splits overlap in **2 documents**. Any "held-out" claim is therefore scoped to the
corpus size it was computed at, and comparing two runs at different sizes compares
two different validation sets.

**What this means for the claim.** The model does not pass the gate. Beyond that,
the honest statement is that **the training pipeline currently cannot tell you
whether it is working**, because the number it reports comes from the in-memory
model while the artifact on disk is a different quantity. Section 18.10 concluded
that token accuracy was the wrong metric; the correction is that the replacement
also needs to be measured on the *saved* artifact, not the live one. Until the
discrepancy is closed, no run's log should be trusted as evidence about a
checkpoint.

### 18.14 The gap was thirty-two floats

Section 18.13 ended with a contradiction it could not explain: the log reported
`loss 0.0129  recall 100.0%` while every checkpoint written measured `loss 3.7` and
generated repetition loops. The question was whether the log or the artifact was
lying.

Neither was. The checkpoint was being corrupted on the way *in*.

`RotaryEmbedding` held its frequencies in
`register_buffer("inv_freq", ..., persistent=False)`. A non-persistent buffer is
not written to the checkpoint. Loading allocates every buffer the module declares
and fills only the ones present in the state dict, so a reloaded model computed
attention positions from uninitialised memory:

```text
trained model : [1.0, 0.6978, 0.4869, 0.3398]
reloaded      : [-9.65e-11, 3.09e-41, 0.0, 0.0]
```

Thirty-two floats. Rebuilding them from `head_dim` and `theta` made the two
forward passes agree exactly -- `max|diff| = 0.0` -- and the checkpoints then
measured what the log had claimed all along:

| checkpoint | before | after | log claimed |
| --- | --- | --- | --- |
| `final` | json 50.0% | json 87.5% | -- |
| `step2000` | json 0.0% | recall 97.9% | recall 96.2% |
| `step2400` | json 0.0% | recall 97.9% | recall 100.0% |
| `step2800` | json 0.0% | recall 99.2%, stp 92% | recall 100.0%, stp 100.0% |

**Why it hid for so long.** Every check that would normally catch a broken
checkpoint passed. All 214 weight tensors matched byte for byte. Every config
field round-tripped. The buffer's name, shape and dtype were correct -- only its
*contents* were absent, and nothing compares contents that were never saved. The
earlier round-trip tests "passed" because they used an undertrained model whose
output was the same garbage either way; a metric comparing categorical rates on a
model that emits nothing agrees at 0% no matter how broken the load is. It took a
token-for-token comparison to expose it.

**The instrument now exists.** Three things were added, and each one is the
answer to a specific way the old pipeline could mislead:

- `evaluate_checkpoint(path, ...)` measures a checkpoint **on disk**. The log's
  numbers came from the in-memory model, and nothing measured the artifact, so
  the two could diverge without a single line of output changing.
- `_verify_saved` runs after every save during training and prints
  `SAVED stepN: ... (live: ...) MATCH|MISMATCH`. A divergence is now visible at
  the step it happens rather than after the run is over.
- `--verify <dir>` answers "what does this artifact actually do?" for any
  checkpoint, including ones written by earlier runs. Its first use is what
  produced the table above.

**Three more bugs found while fixing this.**

`split_examples` assigned documents with `torch.randperm(len(items))`. Two
consequences, both measured. It split *inside* a value-set, and `build_examples`
renders three documents per value-set -- clean, messy, hostile -- so holding out
the clean variant left the same answer in training through its siblings: **600 of
600** validation answers were reachable by recall rather than extraction. And
`randperm` depends on the item count, so the split depended on corpus size; two
runs at different `--docs` compared different validation sets while both claiming
the same seed. Assigning whole value-sets by a hash of their contents fixes both.
Leakage is now 0 and a subset corpus assigns its documents exactly as the full one
does.

`stratified_sample` replaces a prefix slice in two places, and this one is worth
recording because it made a failing model look perfect in *both* directions. The
corpus is difficulty-major -- all clean, then all messy, then all hostile -- so
`examples[:n]` is not a sample, it is the easy end. `generation_metrics` took
`examples[:8]`: **seven clean documents**, reported as `recall 100.0%`. The
acceptance gate took `examples[:30]`: **thirty clean documents**, so the gate never
scored `messy` or `hostile` at all -- the two levels the model card itself calls
the hard case. Both numbers were answers to a narrower question than they appeared
to answer. One shared implementation now lives in `fields.py`, because
`acceptance -> workflow -> model.evaluate_model -> model.train -> acceptance` is a
real cycle and a helper shared by both has to sit below it.

**The honest result.** With the corruption fixed and sampling balanced, on 648
held-out documents from the training corpus the model reaches **99.2% field recall
and 92% document STP**. On `seed=1`, a corpus the run never saw, the acceptance
gate scores it **71.7% field accuracy and 0% document STP**, and reports FAIL.

That gap is real and is the number to work on next. It is also only visible now:
before this section, the gate was sampling one difficulty, the training metric was
sampling one difficulty, and the checkpoints were being silently rewritten on
load. Three independent ways to be told the model was better than it is.

### 18.15 The validation set was inside the region the model had already fitted

Section 18.14 left a gap it could not explain. With the RoPE corruption fixed and
sampling balanced, the model scored **99.2% field recall and 92% document STP** on
648 held-out documents -- and **71.7%** on the acceptance gate's corpus. Same
model, same generator, same difficulty mix. The question was which number
described the model.

Neither, quite. One described the training draw and the other described a fresh
one, and the difference is large enough that the distinction has to be made
explicit.

**Isolating it.** `build_corpus` takes values and rendering separately, so the two
can be crossed:

| values | prose rendered with | field recall |
| --- | --- | --- |
| seed 0 | seed 0 | 97.5% |
| seed 0 | seed 1 | 97.5% |
| seed 1 | seed 0 | 61.2% |
| seed 1 | seed 1 | 62.1% |

The prose rendering is **irrelevant** -- changing it moves nothing. The values are
**everything**. The model is not reading the document; it is recognising values it
has already seen.

**Why the held-out split could not see this.** `generate_many` walks one seeded
stream, so a shorter draw is a prefix of a longer one:

```text
generate_many(60,  seed=0)  subset of  generate_many(4000, seed=0)   -- 60/60
generate_many(400, seed=1)  disjoint from  generate_many(4000, seed=0) --  0/400
```

Training used 4,000 value-sets and the split held out 216 of them. Those 216 are
neighbours of the training ones, drawn from the same pool in the same order, so a
model that stores value instances scores well on them regardless of whether it
learned to extract. 53M parameters can hold 4,000 value-sets of IBANs, amounts and
remittance strings; the within-draw figure measured storage, not skill.

**Ruled out, each by measurement rather than argument.** The label vocabulary is
identical between draws (41 strings, none unique to either). Document structure is
identical (same line counts, same labelled-line counts, same distractor rates).
The US/EU amount-format mix is identical (33.3%/17.1% versus 33.3%/16.5%). Every
value *shape* used by the gate appears in training (0 unseen of 4 shapes for
`PmtId_EndToEndId`, 6 for BIC, 1 for date, 4 for IBAN). And all ten fields are
present in every document of every difficulty in both draws. What differs is the
instances: only 0.5% of the gate's remittance strings and 32.3% of its BICs appear
anywhere in the training corpus.

**What changed.**

`holdout_examples(count, seed)` builds a validation corpus from a **disjoint draw**,
and `--holdout-seed` wires it in. The training loop's own validation slice is
retained for divergence detection -- that is what it is for -- but the generation
metric now scores the disjoint corpus, so the number in the log is a
generalisation number. `--holdout-seed` equal to `--seed` is refused rather than
silently accepted, because that is exactly the configuration that overstated the
model.

Six tests pin the properties: the holdout is disjoint, covers every difficulty, is
deterministic; a short draw is a prefix of a long one; two seeds share no
value-sets; and the generator still emits the canonical key set, so the shapes
these tests rely on cannot drift unnoticed.

**The honest result, measured on the artifact:**

| corpus | field recall | doc STP |
| --- | --- | --- |
| within-draw slice (what was reported before) | 99.2% | 92% |
| disjoint draw (`--holdout-seed`) | **66.3%** | **0%** |

**What this means for the project.** The model does not pass the acceptance gate,
and did not when §18.12 first reported otherwise. The architecture, the corpus
generator and the export are in reasonable shape; what the model lacks is evidence
that it generalises, because it has only ever been trained on one draw of 4,000
value-sets. The next step is not more epochs on seed 0 -- it is more draws, so that
extraction is the only strategy that scores well. The instrumentation to see the
difference now exists and is wired in by default.

### 18.16 A fix that measured as a no-op, and was removed

§18.15 ended with a diagnosis and a proposed remedy: the model had memorised one
draw of 4,000 value-sets, so split the document budget across many draws and
recognition would stop working.

The reasoning was wrong, and measurement said so before any training run was
needed. Comparing corpora at a fixed document count:

```text
config                  docs  uniq text  uniq BIC  uniq Amt  uniq remit
draws=1  docs=1000      3000       3000       483      1000        1000
draws=10 docs=1000      3000       3000       485      1000         999
draws=50 docs=1000      3000       3000       487      1000        1000
```

Split across fifty draws, the corpus contains the same number of distinct BICs,
amounts and remittance strings as it does in one draw. A different *selection*
from the same space is not a larger one. `generate_many` draws without
replacement, so `count` value-sets is `count` value-sets however they are
partitioned; the flag could not have affected memorisation, and a training run
comparing it would have spent about an hour to measure noise.

The flag was implemented, measured, and then **removed**, along with
`draw_seeds` and the `--train-draws` argument the guard used. It is recorded here
because the alternative -- shipping it with a corrected help string, as a flag
that provably does nothing -- is worse than not having it, and because the
mistake is instructive: "make the data more varied" is not a lever, and only
counting the distinct values shows that.

**What actually bounds generalisation.** The number of value-sets. One draw of
4,000 is 4,000 mappings of values to JSON, and 53M parameters store that
comfortably -- which is exactly what the 97.5%-versus-61.2% split measures.
Raising `count` is the only thing in `build_examples` that moves the figure.

**What survives from this attempt.** The guard, which is the part that mattered.
`evaluate_model.py` previously carried a comment saying its documents came from a
different seed from training and so the model had not seen those layouts. That
comment was an assumption doing the work of a check, and it was the assumption
that let a 61%-on-fresh-values model be reported at 97%. Both entry points now
compare the evaluation seed against the training seed and **refuse** rather than
proceed, `--holdout-seed` defaults to a disjoint draw, and six tests pin the
properties. That is the substance of §18.15 and it is unchanged by the removal.

The honest position after all of it: the model reaches **66.3% field recall and
0% document STP** on a fresh draw, the acceptance gate fails, and the way to move
that number is more value-sets -- more data, not better scheduling of the data it
already has.

### 18.17 The gate does not pass, and the numbers that say why

Two training runs were attempted against the acceptance gate. Neither passes.
This section records what each measured, because the comparison that looks
obvious is not the comparison that is valid.

**The gate's verdict, on the model trained the old way** (`steps2800`, 4,000
value-sets, all three difficulty levels, 14 epochs), scored against a fresh value
draw:

```text
[PASS] C1 valid JSON       [PASS] C2 canonical keys   [PASS] C3 no duplicates
[PASS] C9 routing          [PASS] C6 deterministic    [PASS] C7 XSD round-trip
[FAIL] C4 grounded         0.0000   (need >= 0.9800)
[FAIL] C8 role consistency 0.0000   (need >= 1.0000)
[FAIL] C5 field accuracy   0.7167   (need >= 0.9000)
[FAIL] C5b document STP    0.0000   (need >= 0.5000)
VERDICT: FAIL -- do not publish
```

**And the model trained with one difficulty level per value-set** (12,000 distinct
value-sets, one level each, 3 epochs):

```text
[PASS] C1  [PASS] C2  [PASS] C3  [PASS] C9  [PASS] C6  [PASS] C7
[FAIL] C4 grounded         0.0000
[FAIL] C8 role consistency 0.0000
[FAIL] C5 field accuracy   0.2467   (need >= 0.9000)
[FAIL] C5b document STP    0.0000
VERDICT: FAIL -- do not publish
```

**The comparison that is not valid.** 71.7% versus 24.7% reads like the
one-level-per-value-set change made the model four times worse. It did not
demonstrate that, because the two runs did not receive the same amount of
training:

| config | epochs | value-sets | levels | exposures |
| --- | --- | --- | --- | --- |
| levels=3 | 14 | 4,000 | 3 | 168,000 |
| levels=1 | 3 | 12,000 | 1 | 36,000 |

The second run had **4.7x less exposure**. Its loss was still falling when it
stopped (1.59, from 6.92, with no sign of plateau) and its held-out field recall
was still climbing at the last evaluation (8.8% -> 15.0% -> 17.5% -> 20.0% ->
22.5% across steps 200/400/600/800/1000). A run stopped mid-descent is not
evidence about the configuration.

**What the comparison does show.** Removing the repetition cost far more learning
speed than it bought in diversity. The three-level rendering is not only a
memorisation signal -- it is augmentation, and each value-set appearing three
times in three different prose layouts teaches the extraction mapping faster than
one appearance does. Against that, the memorisation it enables is what §18.15
measured. Both effects are real; the run above only demonstrates the first.

**Two bugs were found and fixed by attempting this**, and both would have been
missed by reasoning alone:

- **A device-mismatch crash in the checkpoint verifier.** `_verify_saved` passed
  `device="cpu"` for the live model while a GPU run held it on `cuda:0`. Nothing
  raised at the call; it raised inside `nn.Embedding` as "Expected all tensors to
  be on the same device, but got index is on cpu", and it killed the first run at
  its first checkpoint -- after four hours of training had been paid for.
  `generation_metrics` now derives the device from the model's own parameters, so
  the argument cannot be wrong, and because there is no argument. Three tests pin
  it, including one that hands the function a deliberately wrong device.
- **The held-out corpus was built at three levels and sampled at eight
  documents.** `holdout_examples` built 3,000 value-sets x 3 levels = 9,000
  documents for a metric that reads eight of them. It now builds one level per
  value-set: same difficulty coverage, a third of the generation cost.

**What it would take to pass.** The gap on the second and third checks is not
close. C5 needs 90% field accuracy against 24.7% and 71.7%; C5b needs 50%
document STP against 0% on both. Extrapolating the measured rate
(0.46 s/document on this GPU, batch 8, `grad_accum` 4), bringing the one-level
configuration to the exposure of the fourteen-epoch run is about 14 hours, and
there is no evidence that 168,000 exposures reaches 90% either -- the
fourteen-epoch model already had that and scored 71.7%.

The honest position: **the gate is measuring a real deficiency, both models fail
it, and closing it is a training-scale problem rather than a configuration
problem.** The instrumentation to see the deficiency at all is what this session
produced, and it took four independent measurement faults to be fixed before the
number meant anything.
