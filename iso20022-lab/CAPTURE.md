# Inbound capture: OCR, and when not to use it

This is the design document for `documents.py`, `ingest.py` and `ocr.py` — the
layer that turns an inbound artifact into text.

The question it answers is the one that decides the architecture:

> Does adding OCR make field extraction better or worse?

The intuitive answer — *"OCR reads the document, so it can only help"* — is
wrong, and wrong in the specific way that costs money. This document is the
measurement, not the argument.

---

## 1. The routing table

There are five inbound channels. **OCR is needed by two of them.**

| Channel | What arrives | Needs OCR? |
| --- | --- | --- |
| Email body | plain / HTML text | no |
| Email attachment | route by sniffed type | depends |
| PDF, text layer | extractable characters | **no** |
| PDF, scanned | pixels only | **yes** |
| Portal upload | usually HTML → PDF | no, if the layer is sound |
| EDI (MT101 / pain.001) | already structured | **no — parse it** |
| Fax | pixels only | **yes** |

Treating OCR as the default front door is the most common way to make this
system slow, lossy and expensive for no gain. The measured numbers below are why.

---

## 2. The measurement

Both routes are scored on the same 135 documents, with the same rule engine
(`baseline.RuleExtractor`) — 10 fields per document, 1350 fields:

```text
CAPTURE ROUTE COMPARISON
  rules on exact text        fields  88.96%  STP  52.59%  1201/1350  0.0s
  rules on recognised text   fields  14.67%  STP   0.00%   198/1350  82.4s

RECOGNITION TAX (exact minus recognised)
  field accuracy  +74.30%
  document STP    +52.59%
  wall clock      4296x slower
```

Reproduce with:

```bash
python -m iso20022_lab.measure_capture --docs 45 --seed 7 --ocr tiny
```

**The correct baseline is not "regex versus OCR".** It is:

- **A** — rules on the **exact** text: what a text layer or a typed body gives
- **B** — rules on the **recognised** text: what a fax or a scan gives

A is the ceiling. B is what you get when you point a camera at the document.
**The gap is the recognition tax**, and it is 74 percentage points of field
accuracy and *the entire document STP rate*.

That single result settles the routing table: **OCR is only justified when there
is no text layer.** For a document that already carries exact characters,
recognition can only subtract information. There is no accuracy to be gained,
only lost.

---

## 3. Not all damage is equal

A 74% accuracy loss is not a useful engineering number on its own, because most
of that loss is already *visible* to checks the pipeline has. Attributing damage
by cause is what makes it actionable:

```text
DAMAGE BY CAUSE (recognised route)
  role      73  a real value from this document put in the wrong field
  missing  877  nothing extracted; incomplete, so review sees it
  caught   193  characters changed and a validator rejects it
  silent     9  characters changed and nothing rejects it

  visible to existing checks : 1070
  INVISIBLE to all of them   :   82   <- these reach a payment
```

| Cause | What happened | Who catches it |
| --- | --- | --- |
| `missing` | nothing extracted | completeness check → review |
| `caught` | characters changed, checksum fails | validators → review |
| `role` | a **real** value from this document, wrong field | **nobody** |
| `silent` | characters changed, still validates | **nobody** |

**Only 82 of 1070 errors are invisible.** That is the number worth engineering
against, and it is 7.7% of the damage — not 74%.

### 3.1 Why `role` errors appear at all

Look at what recognition did to a label:

```text
truth:  Beneficiary IBAN: FR32260181590830166131860
OCR:    BaneficiarYIBAN: ER32260181590830166131860
```

Recognition **drops and corrupts whitespace**, so labels merge (`BeneficiaryIBAN`)
or break (`BaneficiarY`). When a label is destroyed the extractor loses its
anchor and falls back to positional guessing — and `baseline.py` documents that
the fallback "is wrong roughly half the time."

So recognition damage **amplifies**: a corrupted label does not merely lose one
field, it collapses role assignment for the whole document. This is why `role`
errors are attributed to recognition even though the value itself is intact.

---

## 4. The finding that matters: BIC has no checksum

All **9 silent errors are the same field**:

```text
        CdtrAgt_BICFI          DEUTDEFF349  ->  DEUTDEEE349
        CdtrAgt_BICFI          BNPAFRPP704  ->  BNPAERPP704
        CdtrAgt_BICFI          BNPAFRPP582  ->  BNPAERPP582
        CdtrAgt_BICFI          ABNANL2A836  ->  ARNANI2A836
        CdtrAgt_BICFI          DEUTDEFF372  ->  DLOTDLTT072
        CdtrAgt_BICFI          ABNANL2A800  ->  ABNANI2A800
        CdtrAgt_BICFI          UNCRITMM     ->  UNCDITMM
        CdtrAgt_BICFI          ABNANL2A836  ->  ABNANL2PD36
        CdtrAgt_BICFI          DEUTDEFF466  ->  DEUTDEEE466
```

**8 of the 9 pass both `bic_valid` and `bic_country_valid`.**

Contrast with IBAN, under the same recognition:

```text
  IBAN fields damaged     : 236
  of those, silent        :   0
  of those, caught/missing: 191
```

**Why the difference is not a coincidence.** IBAN carries mod-97 check digits, so
*any* single-character change breaks the checksum. A BIC is four letters of bank
code plus two letters of country plus two of location plus an optional three of
branch — all alphanumeric, **with no check digit at all**. Change a character and
you get another well-formed BIC.

### 4.1 The corrupted BIC lands on a real country

This is the part worth remembering:

```text
  BNPAFRPP704  ->  BNPAERPP704     FR (France)  ->  ER (Eritrea)
  ABNANL2A836  ->  ARNANI2A836     NL (Netherlands) -> NI (Nicaragua)
```

`bic_country_valid` exists specifically to reject prose that matches the BIC
shape, and it does that job perfectly:

```text
  attached      rejected
  beneficiary   rejected
  instruction   rejected
```

But it gives **zero protection against recognition error**, because the
corrupted value is a *valid BIC for a different country*. Both `ER` and `NI` are
real ISO 3166-1 country codes.

**Two different threat models. One check cannot serve both.** A shape-and-country
check is a defence against *prose being mistaken for a value*. It is not, and
cannot be, a defence against *a value being read wrongly*.

This is the same conclusion PROCESS.md §18.3 reached at the **output** end of the
pipeline — where `instruction` matched the BIC pattern and `RU` is a real country
— arrived at independently here from **input** pixels. §18.11 reproduces it on a
clean render, and §18.12 measures it at scale.

### 4.2 What follows

1. **Never run OCR on a document that has a text layer.** Measured: `NL` → `NI`
   and `FF` → `F` on a *clean, machine-printed* render. There is no upside to buy.
2. **A checksummed field is not a verified field.** A checksum detects
   corruption; it does not detect a *valid value that is wrong*. `DEUTDEEE349`
   is a well-formed BIC for a bank that does not exist.
3. **The unchecksummed fields are the exposure.** Nine silent errors, all BIC.
   A BIC field should be confirmed against the payment's own routing data — the
   counterparty's known BIC, not a pattern.
4. **Recognition amplifies beyond the field it damages.** One broken label
   collapses role assignment for the whole document.

---

## 5. What `documents.py` does about it

**Type comes from magic bytes, never the filename or the MIME header.**

```python
sniff(b"\x89PNG\r\n\x1a\n...", "payment.pdf", "application/pdf")
# -> "image/png"
```

Both declarations lie and the bytes win. Routing a raster file to a text-layer
extractor returns an empty string, which reads as *blank document* rather than
*wrong parser* — a silent failure of exactly the kind above.

The same rule applies to email. A raw RFC 822 message is ASCII, so an
ASCII-check in the sniffer classifies it as `text/plain` — and then the **entire
MIME envelope, base64 attachment payload included, is handed downstream as the
document's text**. `documents.py` detects a header block terminated by a blank
line plus at least two mail-specific header names before the text fallback runs.
A test caught this; the assertion was a result count.

**`has_text_layer` is derived, never declared.** The router takes it as a
parameter, which is right as an interface and dangerous as a default — the two
facts that select the expensive path are the two a caller is most likely to
guess. `extract_pdf` tries to read the layer and reports what happened.

**Rasterise only when there is no text layer.** Rendering a text-layer PDF would
cost time and replace exact characters with recognised ones.

**Absent text is not absent capability.** When no renderer or no OCR backend is
available the result says so explicitly. Returning `""` as though it were
extracted is indistinguishable from a blank document at every later stage.

**Attachments are never concatenated into the body.** A covering note is prose
somebody typed; an attachment is a file. Concatenating them lets a number in the
note be read as a field of the payment — a wrong value rather than a parse error.

---

## 6. Running it

```bash
# Route one file and show the decision, no OCR
python -m iso20022_lab.documents statement.pdf

# Force recognition (the fax path)
python -m iso20022_lab.documents fax.png --ocr tiny

# Measure the recognition tax on your own corpus
python -m iso20022_lab.measure_capture --docs 45 --ocr tiny --json-out /tmp/m.json
```

`--json-out` writes per-field damage with causes, which is what you want when
deciding whether a field needs a checksum or a cross-check.

---

## 7. Dependencies

| Component | Choice | Why |
| --- | --- | --- |
| OCR | PP-OCRv6 via **ONNX Runtime** | no PaddlePaddle; ONNX keeps an on-prem deployment small |
| PDF | **pypdfium2** | text extraction *and* rasterisation in one dependency |
| Email | **stdlib `email`** | reference implementation of RFC 822; decodes transfer encoding, charset and the MIME tree |
| HTML | **regex flattener** | the structure that matters is block boundaries; a DOM adds machinery, not text |

`unstructured` was considered and not used. For the inputs this pipeline accepts
— mail and PDF — the standard library plus one PDF library cover the format
completely. A document library buys breadth across formats that are not accepted
here; it does not buy correctness on the ones that are.

The dictionary is load-bearing. PP-OCRv6 tiny recognition has 6906 classes =
6904 characters + CTC blank + space. Feed it the v5 dictionary (18383) or the v1
dictionary (6623) and it does not error — every index still maps to *a*
character, so it produces confident garbage. `load_dictionary` checks the count
against the model's output dimension and refuses to run on a mismatch.

---

## 8. Closing the residual gap: routing cross-checks

Section 4.2 listed four consequences. The fourth was open:

> A BIC field should be confirmed against the payment's own routing data — the
> counterparty's known BIC, not a pattern.

That is now implemented in `routing.py`, and it is the fix for the silent class —
**not** a better recogniser, because a better recogniser still cannot tell you
whether an unchecksummed field is right.

### 8.1 The asymmetry is the tool

Two measured facts, from §2 and §4:

| field | damaged | silent | why |
| --- | --- | --- | --- |
| `CdtrAcct_IBAN` | 236 | **0** | mod-97 catches every single-character change |
| `CdtrAgt_BICFI` | 9 silent | **9** | no check digit at all |

They describe **the same party**. So:

> The IBAN is checksummed, so its country code is trustworthy.
> The BIC is not, so its country code is only a claim.
> Therefore the trustworthy one can test the untrustworthy one.

No directory, no model, no recogniser change. Just a cross-field comparison:

```text
CdtrAcct_IBAN = FR7630...   (mod-97 verified)
CdtrAgt_BICFI = BNPAERPP704 (Eritrea)
-> the account is French, the bank is claimed to be Eritrean: reject
```

That catches 8 of the 9 measured silent errors — every one where recognition
damaged the country code:

```text
BNPAFRPP704 -> BNPAERPP704    FR -> ER (Eritrea)      now FAILS
ABNANL2A836 -> ARNANI2A836    NL -> NI (Nicaragua)    now FAILS
```

### 8.2 The ninth one needs a directory, and that is stated not hidden

`DEUTDEFF349` read as `DEUTDEEE349` keeps its country. Both are German. Country
agreement passes, and no amount of country checking will see it — mapping `DEUT`
to a German bank code requires the BLZ/BIC table, which this project does not
have and will not invent.

So `routing.py` defines a `RoutingDirectory` protocol with three implementations:

| implementation | checks | use |
| --- | --- | --- |
| `NoDirectory` | country only | the default; says so in every report |
| `ObservedDirectory` | country + institution | learns IBAN→BIC from **confirmed** payments |
| caller-supplied | whatever it provides | the hook for a real registry |

With an `ObservedDirectory` holding one settled correspondence, the ninth case is
caught:

```text
[ok  ] bic_country_matches_iban: creditor BIC country DE agrees with the account country
[FAIL] bic_matches_directory: creditor account is held at DEUTDEFF349 but the
        extraction says DEUTDEEE349 -- same country, different institution
```

`ObservedDirectory` learns only from `observe()` calls — values confirmed by a
human or by settlement. It deliberately has no learn-from-predictions path: a
directory built from a model's own output would rediscover the model's errors as
routing rules, and the check would then confirm them.

### 8.3 UNVERIFIABLE is not PASSED

The design decision that matters most. A BIC with no counterparty IBAN, or one
issued in a country with no directory entry, **has not been checked**. The verdict
is `UNVERIFIABLE`, and it is never folded into `PASSED`:

```text
[ok  ] bic_country_matches_iban: creditor BIC country FR agrees with the account country
[??  ] bic_matches_directory: creditor IBAN is not in the routing directory (none),
        so the bank code cannot be confirmed -- only the country was checkable
2 check(s): 0 failed, 1 unverifiable
```

A report that renders that second line as "passed" manufactures confidence out of
an absence. The count of unverifiable checks is printed precisely so a reader can
see how much of the assurance is real.

### 8.4 Wired into the gate as C9

`acceptance.evaluate` now runs C9 at a threshold of 1.00, for the same reason as
C8: a routing disagreement is a payment to the wrong bank, and unlike a misread
description there is no downstream check that catches it.

**One bug found by writing the tests.** `cross_check` resolved its directory with
`directory or NoDirectory()`. `ObservedDirectory` defines `__len__`, so an *empty*
one is **falsy** — and a configured-but-not-yet-populated directory was silently
replaced by "no directory configured". Nothing failed loudly: the check still ran
and still said unverifiable. It just attributed the result to the wrong cause,
which is the difference between an operator reading *"add a directory"* and *"your
directory is empty"*. The fix is `is None`, and the general rule is that a
truthiness test on a caller-supplied collaborator is a silent substitution waiting
to happen.
