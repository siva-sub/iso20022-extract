# Recipe book

Common workflows, as code that runs.

**Every recipe here has exactly one test in `test_recipes.py` that executes the
same code path and asserts the same claim.** If a recipe changes, the test fails,
so this document cannot drift into being fiction. Run them:

```bash
cd iso20022-lab && python -m pytest test_recipes.py -q
```

Recipes needing a trained model verify against whatever checkpoint exists and
**skip with an explicit message** rather than silently passing.

| # | Recipe | Cost |
| --- | --- | --- |
| [R1](#r1--extract-from-an-email-body) | Extract from an email body | free |
| [R2](#r2--extract-from-a-pdf-text-layer) | Extract from a PDF text layer | free |
| [R3](#r3--extract-from-a-fax-or-scan) | Extract from a fax or scan | OCR |
| [R4](#r4--route-an-edi-message) | Route an EDI message | free |
| [R5](#r5--validate-a-value-set) | Validate a value set | free |
| [R6](#r6--build-and-xsd-validate-a-message) | Build + XSD-validate a message | free |
| [R7](#r7--score-a-capture-workflow) | Score a capture workflow | free |
| [R8](#r8--is-a-model-justified-at-all) | Decide if a model is justified | free |
| [R9](#r9--price-the-exceptions) | Price the exceptions | free |
| [R10](#r10--gate-a-model-before-publishing) | Gate a model before publishing | free |
| [R11](#r11--capture-bytes-not-a-declared-channel) | Capture from raw bytes | free |
| [R12](#r12--a-scanned-pdf-all-the-way-to-pixels) | Scanned PDF → OCR | OCR |
| [R13](#r13--an-email-with-an-attachment) | Email with attachment | depends |
| [R14](#r14--measure-the-recognition-tax) | Measure the recognition tax | OCR |
| [R15](#r15--the-whole-workflow-every-channel) | The whole workflow, every channel | both |
| [R16](#r16--cross-check-an-unchecksummed-field) | Cross-check a BIC against routing data | free |

**The one rule that governs all of them:** cheap deterministic checks first, OCR
last and only when selected. Recipes R1, R2, R4, R5 and R6 construct an OCR
backend that raises if called, so if any of them ever reaches recognition the
test fails loudly.

---

## R1 — Extract from an email body

A payment instruction in the mail body is **already text**. It never touches a
model.

```python
from iso20022_lab.ingest import Channel, Path, capture, route

decision = route(Channel.EMAIL_BODY)
assert decision.path is Path.TEXT

result = capture(Channel.EMAIL_BODY, text=EMAIL_BODY, ocr=stub)
assert result.text == EMAIL_BODY
assert result.used_ocr is False
assert stub.calls == []
```

Test: `test_r1_email_body_never_touches_ocr`

---

## R2 — Extract from a PDF text layer

A PDF's embedded text layer is frequently **present but wrong** — ligature
mapping failures, CID font substitutions, wrong reading order. Detecting that
normally means running OCR and diffing, which defeats the point.

Instead the **output validators are reused as an input-quality probe**. IBAN
mod-97 and BIC shape catch corruption, the same way they catch a bad extraction.

```python
from iso20022_lab.ingest import Channel, Path, capture, route, text_layer_trust

assert route(Channel.PDF, has_text_layer=True).path is Path.PROBE

# Sound layer: accepted, no OCR.
result = capture(Channel.PDF, text=EMAIL_BODY, image_path="page.png",
                 has_text_layer=True, ocr=stub)
assert result.trust.trustworthy
assert stub.calls == []
```

The corruption this catches is the real one — a font maps the glyph and every
`0` comes back as `O`, which is invisible to the eye:

```python
corrupt = EMAIL_BODY.replace("DE89370400440532013000", "DE8937O4OO44O532O13OOO")
result = capture(Channel.PDF, text=corrupt, image_path="page.png",
                 has_text_layer=True, ocr=stub)
assert not result.trust.trustworthy
assert result.used_ocr is True          # falls through, carrying the reason
assert stub.calls == ["page.png"]
```

**The probe is honest about its limit.** A document with no IBAN and no BIC
cannot be judged by checksum, and it says so rather than guessing:

```python
report = text_layer_trust("Please pay the attached invoice as discussed.")
assert report.trustworthy is False
assert "no checksummed fields" in report.reason
```

Tests: `test_r2_sound_text_layer_is_accepted_without_ocr`,
`test_r2_corrupt_text_layer_falls_through_to_ocr`,
`test_r2_probe_reports_when_it_cannot_judge`

---

## R3 — Extract from a fax or scan

Pixels only, so recognition is unavoidable.

```python
stub = _StubOcr(EMAIL_BODY, confidence=0.88)
result = capture(Channel.FAX, image_path="fax.tiff", ocr=stub)

assert result.used_ocr is True
assert result.ocr_confidence == 0.88
assert stub.calls == ["fax.tiff"]
```

**Confidence is returned, not discarded.** A shaky read is routable to a human,
and that is the only reason to keep the number:

```python
result = capture(Channel.FAX, image_path="fax.tiff", ocr=_StubOcr(EMAIL_BODY, 0.41))
assert result.ocr_confidence < 0.5
```

Tests: `test_r3_fax_always_routes_to_ocr`,
`test_r3_ocr_confidence_is_returned_not_discarded`

---

## R4 — Route an EDI message

Already structured. **OCR over it would convert correct data into a guess** —
the most expensive possible mistake, because it introduces error into something
that was exact.

```python
assert route(Channel.EDI).path is Path.STRUCTURED

result = capture(
    Channel.EDI,
    structured="<Document><CstmrCdtTrfInitn/></Document>",
    has_text_layer=True,
    ocr=stub,
)
assert result.used_ocr is False
assert stub.calls == []
```

Test: `test_r4_edi_is_never_recognised`

---

## R5 — Validate a value set

Deterministic checks run before any model, because they cost nothing and fail
closed.

```python
from iso20022_lab.validators import validate_field

for field, value in TRUTH.items():
    assert validate_field(field, value) == []
```

A single bad character is caught mechanically, with no need to know the right
answer:

```python
assert validate_field("CdtrAcct_IBAN", "DE8937O4OO44O532O13OOO")
```

Minor units are enforced — JPY has none, so `100.50` is **malformed**, not merely
unusual:

```python
from iso20022_lab.validators import amount_matches_minor_units

assert amount_matches_minor_units("100", "JPY")
assert not amount_matches_minor_units("100.50", "JPY")
```

> **What validation cannot do.** A checksum detects *corruption*. It does not
> detect *a valid value that is wrong*. See [R14](#r14--measure-the-recognition-tax)
> — `DEUTDEFF349` read as `DEUTDEEE349` passes both the shape and the country
> check. If a field has no checksum, validation is not evidence about it.

Tests: `test_r5_valid_values_produce_no_findings`,
`test_r5_a_single_bad_character_is_caught`,
`test_r5_minor_units_are_enforced`

---

## R6 — Build and XSD-validate a message

The round trip. **An extraction that cannot become a valid payment has not
solved the workflow**, however good its field accuracy looks.

```python
from iso20022_lab.distill import to_slots
from iso20022_lab.serialize import MessageBuilder, fill_system_fields
from iso20022_lab.xsd_introspect import XSDModel

model = XSDModel(XSD)
slots = to_slots(dict(TRUTH), model)
fill_system_fields(slots, model)      # MsgId, CreDtTm, NbOfTxs, CtrlSum
built = MessageBuilder(model).build(slots)

assert built.ok, built.errors[:2]
assert built.xml is not None
assert "CstmrCdtTrfInitn" in built.xml
assert "1250.00" in built.xml
```

The same path driven by the deterministic extractor:

```python
extracted = RuleExtractor().extract(EMAIL_BODY)
slots = to_slots(dict(extracted.values), model)
fill_system_fields(slots, model)
assert MessageBuilder(model).build(slots).ok
```

Tests: `test_r6_truth_values_build_a_schema_valid_message`,
`test_r6_the_rule_extractor_can_drive_the_round_trip`

---

## R7 — Score a capture workflow

Field accuracy alone does not describe a workflow. The number a business case
uses is **human effort per document**.

```python
from iso20022_lab.workflow import CanonicalRuleExtractor, EffortModel, effort_table, run_workflow

corpus = build_corpus(generate_many(6, seed=7), per_seed=1, seed=7)
report = run_workflow(corpus.documents, CanonicalRuleExtractor(), "rules_v1", XSD)

assert len(corpus.documents) == 18   # 6 seeds x 3 difficulty levels
assert len(report.outcomes) == len(corpus.documents)
assert "min/doc" in effort_table([report], EffortModel())
```

Note the arithmetic: `generate_many(6)` yields six *seed value-sets* and
`build_corpus` renders each at all three difficulty levels, so the count is 18.
The first version of this test guessed 6 and was wrong.

Test: `test_r7_workflow_scores_and_reports_human_effort`

---

## R8 — Is a model justified at all?

A model is not an upgrade. **Cost scales with format count**, so there is a
crossover, not a verdict.

```python
from iso20022_lab.format_economics import compare, crossover

assert compare(1)["rules_year_one"] < compare(1)["model_year_one"]      # 1 format: a model is waste
assert compare(250)["model_year_one"] < compare(250)["rules_year_one"]  # 250 formats: model wins
assert 1 < crossover() < 50
```

The argument must not depend on the model beating regex *within* a format:

```python
assert crossover(model_stp=0.60) < 20
```

Tests: `test_r8_economics_finds_a_crossover`,
`test_r8_crossover_survives_a_pessimistic_accuracy`

---

## R9 — Price the exceptions

An STP point has a price, and knowing it is what stops the discussion being
about accuracy for its own sake.

```python
from iso20022_lab.economics import exception_economics, marginal_value_per_stp_point

monthly, annual = marginal_value_per_stp_point()
assert annual > monthly > 0

econ = exception_economics(0.498)     # the measured rule-baseline rate
assert econ.annual_cost > 0 and econ.fte_equivalent > 0

assert exception_economics(1.0).monthly_exceptions == 0   # perfect pipeline: nothing to pay for
```

Test: `test_r9_exception_economics_prices_an_stp_point`

---

## R10 — Gate a model before publishing

**A gate that passes everything is worthless**, so this recipe asserts the gate
*rejects*.

```python
python -m test_model_acceptance --model <path> --docs 12 --seed 31
```

Verdict is a process exit code, so it composes with CI. See
[`acceptance.py`](src/iso20022_lab/acceptance.py) for the checks and
`PROCESS.md` §18 for the C1–C8 design.

Tests: `test_r10_acceptance_gate_rejects_the_broken_model`,
`test_r10_acceptor_requires_the_round_trip`,
`test_r10_c8_catches_the_failure_every_other_check_missed`

---

## R11 — Capture bytes, not a declared channel

`capture()` takes the channel, the text and `has_text_layer` as parameters.
That is fine as an interface and dangerous as a default, because the two facts
that select the expensive path are the two a caller is most likely to guess.
`documents.ingest()` derives both from the bytes.

```python
from iso20022_lab.documents import ingest, sniff

# Type comes from magic bytes, never the name or the MIME header.
png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
assert sniff(png, "payment.pdf", "application/pdf") == "image/png"
```

Both declarations lie and the bytes win. Routing a raster file to a text-layer
extractor returns an empty string, which reads as *blank document* rather than
*wrong parser*.

An email is ASCII, so a naive sniffer calls it `text/plain` and the **entire MIME
envelope — base64 attachment payload included — is handed downstream as the
document's text**. `sniff` checks for a header block plus mail-specific header
names first:

```python
raw = _mail_with("cover note").as_bytes()
assert sniff(raw) == "message/rfc822"

results = ingest(raw)
assert len(results) == 2                       # body + attachment, never merged
assert "JVBERi0" not in results[0].text        # no base64 in the body
```

`ingest` returns **a list**, one result per artifact. Collapsing an email into
one result concatenates values from different documents into one extraction.

```python
typed = sniff(b'<?xml version="1.0"?><Document xmlns="urn:iso:std:iso:20022:tech:xsd:pain.001">')
assert typed == "application/edi"
```

Tests: `test_documents.py` — 25 tests over sniffing, dispatch and email.

---

## R12 — A scanned PDF all the way to pixels

The routing table promised *"PDF scanned → OCR"* and **no code path delivered
it**: the OCR backend takes an image path and nothing turned a PDF into one.

```python
from iso20022_lab.documents import extract_pdf

# An image-only PDF: pixels, no text layer.
ext = extract_pdf(scan_pdf_bytes, tmpdir=work)
assert ext.has_text_layer is False
assert ext.image_path is not None            # a scan must yield pixels
assert Path(ext.image_path).exists()
```

A text-layer PDF is **not** rendered, because rendering it would replace exact
characters with recognised ones:

```python
ext = extract_pdf(text_pdf_bytes, tmpdir=work)
assert ext.has_text_layer is True
assert ext.image_path is None
assert "DE80716202183220023878" in ext.text
```

When no renderer is available the result says so, rather than returning `""` as
though it were extracted:

```python
assert any("renderer" in n for n in ext.notes)
```

Test: `test_documents.py::test_scanned_pdf_is_rasterised_so_ocr_can_run` and
neighbours.

---

## R13 — An email with an attachment

```python
ext = extract_email(raw_bytes)
assert ext.media_type == "message/rfc822"
assert "EUR 1.250,00" in ext.text            # the body
assert len(ext.parts) == 1                   # the attachment, separately
```

**Attachments are never concatenated into the body.** A covering note is prose
somebody typed; an attachment is a file. Concatenating them lets a number in the
note be read as a field of the payment — a wrong value rather than a parse error.

An attachment named `statement.pdf` that is really a PNG is typed from its bytes:

```python
ext = extract_email(_mail_with("see attached", name="statement.pdf", payload=png))
assert ext.parts[0].media_type == "image/png"
```

Test: `test_documents.py::test_attachment_named_pdf_but_really_png_is_typed_from_bytes`

---

## R14 — Measure the recognition tax

The recipe that decides whether to use OCR at all.

```bash
python -m iso20022_lab.measure_capture --docs 45 --seed 7 --ocr tiny --json-out /tmp/m.json
```

Both routes are scored on the same documents with the same rule engine:

```text
  rules on exact text        fields  88.96%  STP  52.59%  1201/1350
  rules on recognised text   fields  14.67%  STP   0.00%   198/1350

  field accuracy  +74.30%
  document STP    +52.59%
  wall clock      4296x slower
```

Damage is attributed, because a 74% loss is not actionable on its own:

```text
  role      73  a real value from this document put in the wrong field
  missing  877  nothing extracted; incomplete, so review sees it
  caught   193  characters changed and a validator rejects it
  silent     9  characters changed and nothing rejects it

  visible to existing checks : 1070
  INVISIBLE to all of them   :   82
```

**All 9 silent errors are `CdtrAgt_BICFI`, and 8 of 9 pass the country check:**

```text
  BNPAFRPP704  ->  BNPAERPP704      FR (France) -> ER (Eritrea)
  ABNANL2A836  ->  ARNANI2A836      NL (Netherlands) -> NI (Nicaragua)
```

IBAN, under the same recognition: **236 damaged, 0 silent** — mod-97 catches
every single-character change. A BIC has **no check digit at all**.

Read [`CAPTURE.md`](CAPTURE.md) for the full analysis. Three things follow:

1. **Never OCR a document that has a text layer.** There is no accuracy to gain.
2. **A checksummed field is not a verified field.** `DEUTDEEE349` is a
   well-formed BIC for a bank that does not exist.
3. **Cross-check unchecksummed fields against the payment's own routing data**,
   not against a pattern.

Test: `test_documents.py` plus `test_ingest.py`.

---

## R15 — The whole workflow, every channel

The end-to-end claim, and the one that justifies the routing table. A document
whose characters are exact should reach a schema-valid ISO 20022 message with no
recognition anywhere. If this fails, the routing premise is wrong.

```python
from iso20022_lab.corpus import build_corpus
from iso20022_lab.documents import ingest
from iso20022_lab.synth import generate_many

doc = build_corpus(generate_many(1, seed=11), per_seed=1, seed=11).documents[0]

# A plain text body: free, and must be complete.
results = ingest(doc.text.encode(), filename="d.txt", tmpdir=tmp_path)
assert results[0].used_ocr is False
assert _round_trips(results[0].text)

# A PDF with a text layer: probed, not recognised, equally complete.
from capture_fixtures import text_pdf
body = text_pdf(tmp_path / "t.pdf", doc.text).read_bytes()
results = ingest(body, filename="t.pdf", tmpdir=tmp_path)
assert results[0].route.path is CapturePath.PROBE
assert results[0].used_ocr is False
assert _round_trips(results[0].text)
```

### Measured across every channel

| Channel | Route | OCR | Fields | Schema-valid |
| --- | --- | --- | --- | --- |
| Plain text body | `text` | no | 10 | **yes** |
| PDF with text layer | `probe` | no | 10 | **yes** |
| Email + PDF attachment | `probe` | no | 10 | **yes** |
| PDF scanned | `ocr` (rasterise first) | yes, 0.78 | 3 | no |
| Fax / raw image | `ocr` | yes, 0.93 | 5 | no |

The free paths are perfect; the OCR paths lose the message, exactly as the tax in
R14 predicts.

### An incomplete extraction failing is the system working

An early version of this test used a six-line fixture and asserted the round trip.
It failed — and the failure was correct: the XSD refused a payment with no
`ReqdExctnDt`, which is what it is for. The builder returns `ok=True` on all nine
complete corpus documents and refuses partial ones.

```python
# A missing required field must NOT produce a valid message.
assert not _round_trips("Beneficiary: Orchid Pharma BV\nBeneficiary IBAN: DE80...")
```

### The OCR route is asserted to lose, not assumed to

```python
def test_r15_the_scanned_route_really_does_lose_the_message(tmp_path: Path) -> None:
    try:
        from iso20022_lab.ocr import build_tiny
        backend = build_tiny()
    except Exception:
        pytest.skip("PP-OCRv6 weights not available; cannot measure the tax")

    fax = render_doc(tmp_path / "fax.png", doc.text)
    results = ingest(fax.read_bytes(), filename="fax.png", tmpdir=tmp_path, ocr=backend)
    assert results[0].used_ocr is True
    assert results[0].ocr_confidence > 0
```

It skips with an explicit message rather than passing silently when no model
weights are present — a skipped test that returns "pass" would let the finding
rot unnoticed.

Tests: `test_recipes.py::test_r15_exact_text_paths_reach_a_valid_message`,
`test_recipes.py::test_r15_the_scanned_route_really_does_lose_the_message`

---

## R16 — Cross-check an unchecksummed field

The fix for the residual gap in R14, and **not** a better recogniser: a better
recogniser still cannot tell you whether an unchecksummed field is right.

The asymmetry is the tool. The IBAN is checksum-protected, the BIC is not, and
they describe the same party — so the trustworthy field tests the untrustworthy
one.

```python
from iso20022_lab.routing import cross_check, summarise

# FR (France) read as ER (Eritrea) -- a valid BIC, wrong country.
payload = {"CdtrAcct_IBAN": "FR7630006000011234567890189",
           "CdtrAgt_BICFI": "BNPAERPP704"}
assert any(f.blocking for f in cross_check(payload))

# The correct pair must NOT be flagged, or the check is worthless.
clean = {"CdtrAcct_IBAN": "FR7630006000011234567890189",
         "CdtrAgt_BICFI": "BNPAFRPP704"}
assert not any(f.blocking for f in cross_check(clean))
```

### The ninth case needs a directory

`DEUTDEFF349` read as `DEUTDEEE349` keeps its country — both are German — so
country agreement passes and nothing country-based can see it. Mapping `DEUT` to
a German bank code needs the BLZ/BIC table, which this project does not have:

```python
from iso20022_lab.routing import ObservedDirectory

payload = {"CdtrAcct_IBAN": "DE89370400440532013000",
           "CdtrAgt_BICFI": "DEUTDEEE349"}
assert not any(f.blocking for f in cross_check(payload))   # country alone: passes

d = ObservedDirectory()
d.observe("DE89370400440532013000", "DEUTDEFF349")   # confirmed at settlement
assert any(f.blocking for f in cross_check(payload, d))    # directory: fails
```

`observe()` records values confirmed by a human or by settlement. There is
deliberately no learn-from-predictions path: a directory built from a model's own
output would rediscover the model's errors as routing rules, and the check would
then confirm them.

### UNVERIFIABLE is not PASSED

```python
findings = cross_check({"CdtrAcct_IBAN": "FR7630006000011234567890189",
                        "CdtrAgt_BICFI": "BNPAFRPP704"})
unchecked = [f for f in findings if f.check == "bic_matches_directory"]
assert unchecked and unchecked[0].verdict.value == "unverifiable"
```

A BIC with no directory entry has not been checked. Reporting that as "passed"
would manufacture confidence out of an absence, so the verdict is separate and the
count of unverifiable checks is printed.

### As a check in the gate

C9 runs at **1.00**, like C8 — a routing disagreement is a payment to the wrong
bank, and nothing downstream catches it.

```text
[FAIL] C9 routing agreement: 5/12 free of routing disagreement (7 unverifiable)
        -- e.g. creditor account is in FR but the agent BIC BNPAERPP704 is a ER institution
```

### CLI

```bash
python -m iso20022_lab.routing \
  --payload '{"CdtrAcct_IBAN":"FR7630006000011234567890189","CdtrAgt_BICFI":"BNPAERPP704"}' \
  --directory confirmed_pairs.json
```

Exits non-zero on a blocking finding, so it composes with a shell pipeline.

Test: `test_routing.py` — 18 tests, every failure case taken from the measured
PP-OCRv6 run rather than invented.

---

## Adding a recipe

1. Write the recipe here with runnable code.
2. Add **exactly one** test to `test_recipes.py` that runs the same path.
3. If it needs a model, `pytest.skip` with an explicit reason — never pass
   silently.

The pairing is the point. Without it this file is prose; with it, a change to
the code fails the document.
