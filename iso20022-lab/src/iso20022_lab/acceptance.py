"""Acceptance primitives: what "the model works" means, as code.

These live in the library rather than in a test file because they are not
test-only. The acceptance gate uses them to decide whether an artifact is
publishable, and the recipe book uses them to demonstrate the gate. Both import
from here.

That is a deliberate structural choice. The first version had the recipes import
`test_model_acceptance` directly, which worked only when the test directory
happened to be on the type-checker's path -- a module-resolution accident rather
than a design. Test files importing each other is fragile for the same reason in
any project, and it hides the fact that the logic was reusable all along.

THE FAILURE MODES ARE DRAWN FROM REAL OUTPUT

These checks exist in this shape because of what the step-400 checkpoint actually
produced:

    {"Amt_InstdAmt":"47Amt_Ccy":"EUR","CdtrAcct_Ccy":"EUR","CdtrAgt_Ccy":"DE9",
     "Cdtr_Nm":"DE9","Cdtr_Nm":"Orchid Pharma BV"}

One sample, four separate violations: invented keys, a repeated key, values that
appear nowhere in the source document, and a missing close. A suite written from
the specification would have caught none of them, because the specification does
not describe how a half-trained model fails.
"""

# torch re-exports its public C symbols through a private module, so
# `torch.tensor` is reported as a non-public import. It is the documented API
# and the flag is the known false positive. Same directive as model/train.py,
# which hit this first; the project config also disables the check.
# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type-only, so this module stays dependency-light at runtime and does not
    # pull in the corpus and synthesis stack just to name a parameter type.
    from iso20022_lab.corpus import Document

# The canonical field space and the parsers for it live in `fields`, which
# imports nothing from this package. They were defined here, and the training
# metric importing them from here closed an import cycle:
#
#     acceptance -> workflow -> model.evaluate_model -> model.train -> acceptance
#
# Re-exported rather than re-defined so existing callers keep working, and so
# there is exactly one implementation of each.
from iso20022_lab.fields import (
    CANONICAL_KEYS,
    count_duplicate_keys,
    parse_strict,
    stratified_sample,
)
from iso20022_lab.routing import Verdict, cross_check

# Thresholds, as named constants so a failing run names the bar it missed rather
# than leaving a reviewer to infer what "good" was supposed to be.
MIN_JSON_PARSE_RATE = 1.00
MIN_CLEAN_KEY_RATE = 1.00
MIN_GROUNDED_RATE = 0.98
MIN_FIELD_ACCURACY = 0.90
MIN_STP_RATE = 0.50
MIN_XSD_PASS_RATE = 0.90
# C8. Set at 1.00 rather than a tolerance: a role inversion is a wrong recipient,
# and unlike a misread field there is no downstream check that catches it. The
# XSD passed the exact extraction this check exists to reject.
MIN_ROLE_CLEAN_RATE = 1.00
# C9. Also 1.00, for the same reason as C8 and one more: this is the check that
# catches the residual class C8 cannot reach. C8 rejects a BIC that is not
# anchored to a label in the source; it cannot reject a BIC that was anchored and
# then misread, because a misread BIC is still a well-formed BIC -- often for a
# different real country. A routing mismatch is a payment to the wrong bank, and
# nothing downstream catches it.
MIN_ROUTING_CLEAN_RATE = 1.00


@dataclass
class Check:
    """One acceptance check and its verdict."""

    name: str
    passed: bool
    detail: str
    value: float | None = None
    threshold: float | None = None

    def line(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        if self.value is None:
            return f"  [{mark}] {self.name}: {self.detail}"
        return (
            f"  [{mark}] {self.name}: {self.value:.4f} "
            f"(need >= {self.threshold:.4f}) -- {self.detail}"
        )


@dataclass
class AcceptanceReport:
    checks: list[Check] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)
    docs: int = 0

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def render(self) -> str:
        lines = [
            "=" * 78,
            f"MODEL ACCEPTANCE GATE -- {self.docs} held-out documents",
            "=" * 78,
        ]
        lines.extend(c.line() for c in self.checks)
        lines.append("")
        verdict = "PASS -- publishable" if self.passed else "FAIL -- do not publish"
        lines.append(f"VERDICT: {verdict}")
        if not self.passed:
            failed = [c.name for c in self.checks if not c.passed]
            lines.append(f"blocking: {', '.join(failed)}")
        if self.samples:
            lines.append("")
            lines.append("failing samples (first 3):")
            lines.extend(f"  {s}" for s in self.samples[:3])
        return "\n".join(lines)


def normalise(s: str) -> str:
    """Comparison form: case- and separator-insensitive.

    A model may legitimately normalise a value -- reordering a date, dropping an
    IBAN's spaces. It may not invent one. Normalising before comparison is what
    separates those two cases, so that reformatting is not scored as
    hallucination.
    """
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


# --------------------------------------------------------------------------
# C8 -- role consistency
# --------------------------------------------------------------------------

# A role label and the value that follows it. Deliberately tolerant of the
# spacing a recogniser produces: the real OCR run emitted `Debtor:Acme GmbH`
# with no space after the colon, and a stricter pattern would have silently found
# no roles at all and reported the document as unverifiable rather than wrong.
_ROLE_LINE = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z \t]{2,24}?)\s*[:#]\s*(?P<value>[^\n]+)",
)

# Which side of the payment a label belongs to. `ordering`/`payer` are the
# debtor's synonyms and `beneficiary`/`payee` the creditor's, because section
# 16.3 showed the corpus uses synonyms and a check that only knew the canonical
# words would be blind on exactly the messy documents that matter.
_DEBTOR_WORDS = ("debtor", "payer", "ordering", "orderer", "remitter")
_CREDITOR_WORDS = ("creditor", "beneficiary", "payee", "recipient")

_ACCOUNT_FIELDS = ("DbtrAcct_IBAN", "CdtrAcct_IBAN", "DbtrAcct", "CdtrAcct")

# A BIC label and the token after it, mirroring the capture probe's rule in
# section 18.3. Shape alone is not evidence: `INSTRUCTION` and `attached` both
# match the BIC pattern and both have a real ISO country code at positions 5-6.
_BIC_LABELLED = re.compile(
    r"(?:\bBIC\b|\bBICFI\b|\bSWIFT\b|BANK\s+IDENTIFIER|IDENTIFIER\s+CODE)"
    r"\s*[:#]?\s*([A-Za-z0-9]{8,11})\b",
    re.IGNORECASE,
)


def _bic_labelled_values(text: str) -> set[str]:
    """Values sitting directly behind a BIC label in the source."""
    return {normalise(m.group(1)) for m in _BIC_LABELLED.finditer(text)}


@dataclass
class RoleFinding:
    """One role or provenance problem found in an extraction."""

    field: str
    value: str
    problem: str

    def __str__(self) -> str:
        return f"{self.field}={self.value!r}: {self.problem}"


def _labelled_roles(text: str) -> tuple[set[str], set[str]]:
    """Values appearing on debtor-labelled and creditor-labelled lines.

    Values are normalised before comparison, and a line's value has any trailing
    parenthetical or punctuation stripped, because recognisers add and drop both.
    """
    debtor_vals: set[str] = set()
    creditor_vals: set[str] = set()
    for m in _ROLE_LINE.finditer(text):
        label = m.group("label").strip().lower()
        value = normalise(m.group("value"))
        if not value:
            continue
        if any(w in label for w in _DEBTOR_WORDS):
            debtor_vals.add(value)
        elif any(w in label for w in _CREDITOR_WORDS):
            creditor_vals.add(value)
    return debtor_vals, creditor_vals


def role_consistency(text: str, payload: dict[str, str]) -> list[RoleFinding]:
    """Check an extraction against the document's OWN role labels.

    WHY THIS EXISTS, AND WHY IT IS NOT C5. Every other check in this module passed
    on a real OCR run that produced a wrong payment:

        CdtrAcct_IBAN   'DE89370400440532013000'   <- the DEBTOR's account
        DbtrAcct_IBAN   'GB29NWBK60161331926819'   <- the CREDITOR's
        Amt_InstdAmt    '29.00'                    <- should be 1250.00
        XSD valid: True

    C1 through C4 all passed: the values are well-formed, use canonical keys, are
    unique, and are traceable to the document. C7 passed because the XSD checks
    structure and datatypes, not provenance. And C5 is accuracy against ground
    truth, which is not available at run time.

    This check needs no ground truth. It compares the extraction against the
    source document's own labels, which is the one authority present at inference
    time. If the value placed in the creditor's account slot appears in the
    document under a debtor label, the roles are inverted, and that is true
    regardless of whether either IBAN is valid -- which both were.

    Deliberately checks only what can be established from the text. It cannot
    tell that a value is wrong; only that it is in the wrong place.
    """
    findings: list[RoleFinding] = []
    debtor_labelled, creditor_labelled = _labelled_roles(text)

    def val(field: str) -> str:
        for k, v in payload.items():
            if k == field or k.endswith(field):
                return normalise(v)
        return ""

    dbtr_acct = val("DbtrAcct_IBAN")
    cdtr_acct = val("CdtrAcct_IBAN")

    # 8a -- the two sides must be different accounts.
    if dbtr_acct and cdtr_acct and dbtr_acct == cdtr_acct:
        findings.append(
            RoleFinding(
                "DbtrAcct_IBAN",
                dbtr_acct,
                "debtor and creditor accounts are identical (self-payment)",
            )
        )

    # 8b -- a role's value must not be the value the document labels the other way.
    if cdtr_acct and cdtr_acct in debtor_labelled and cdtr_acct not in creditor_labelled:
        findings.append(
            RoleFinding(
                "CdtrAcct_IBAN",
                cdtr_acct,
                "value appears under a DEBTOR label in the source; roles are inverted",
            )
        )
    if dbtr_acct and dbtr_acct in creditor_labelled and dbtr_acct not in debtor_labelled:
        findings.append(
            RoleFinding(
                "DbtrAcct_IBAN",
                dbtr_acct,
                "value appears under a CREDITOR label in the source; roles are inverted",
            )
        )

    # 8c -- the amount must come from an AMOUNT-LIKE position, not merely appear
    # somewhere in the document. The real run produced `29.00`, spliced from the
    # `29` of `GB29NWBK...` and the `.00` of `1.250.00`.
    #
    # An earlier version of this check tested whether the amount was a substring
    # of an account identifier. It never fired, and the premise was wrong: a
    # splice is not a substring, which is precisely why it is hard to see. What
    # the splice does fail is provenance -- `2900` appears nowhere in the source
    # under an amount context, so this tests for that instead. C4 catches the same
    # value as ungrounded; this reports it as a field-boundary failure, which is
    # the actionable diagnosis.
    amount = val("Amt_InstdAmt")
    if amount:
        amount_context = re.compile(
            r"(?:amount|amt|total|sum|value|betrag|montant|importe)[^\n]{0,40}"
            r"|(?:EUR|USD|GBP|CHF|JPY|SEK|NOK|DKK|PLN|CZK)\s*[\d.,]+",
            re.IGNORECASE,
        )
        ctx_hay = "".join(normalise(m.group(0)) for m in amount_context.finditer(text))
        if amount not in ctx_hay:
            findings.append(
                RoleFinding(
                    "Amt_InstdAmt",
                    amount,
                    "value does not appear in the source in an amount context, so "
                    "it was assembled from unrelated fields rather than read",
                )
            )

    # 8d -- a BIC field must be anchored to a BIC label in the source, not merely
    # BIC-shaped. The real run put `INSTRUCTION` in CdtrAgt_BICFI, and that value
    # passes bic_valid AND bic_country_valid, because `RU` at positions 5-6 is
    # Russia. Shape plus a real country code is not enough -- the same conclusion
    # the capture probe reached in section 18.3, for the same reason.
    bic = val("CdtrAgt_BICFI")
    if bic and bic not in _bic_labelled_values(text):
        findings.append(
            RoleFinding(
                "CdtrAgt_BICFI",
                bic,
                "value is not anchored to a BIC label in the source, so it is an "
                "ordinary word that happens to match the BIC shape",
            )
        )

    return findings


def roundtrip_check(
    documents: list[Document],
    model_dir: object,
    tokenizer_dir: object,
    xsd: object,
) -> Check:
    """C7: can the extraction become a valid ISO 20022 message?

    This is the bar that matters. PROCESS.md section 16.3: an extraction that
    cannot become a valid payment has not solved the workflow, however good its
    field accuracy looks. The other checks are necessary conditions; this is the
    point of the exercise.

    Delegates to the workflow harness rather than re-implementing message
    building, so the acceptance gate and the business measurement score the model
    through identical code. Two harnesses would produce two numbers for one
    model, and the discrepancy would be the harness.

    Imports are inside the function because `workflow` pulls in the message
    builder and schema machinery, and this module is otherwise dependency-light.
    """
    from pathlib import Path

    from iso20022_lab.workflow import ModelExtractor, run_workflow

    xsd_path = Path(str(xsd))
    if not xsd_path.exists():
        return Check("C7 XSD round-trip", False, f"schema not found: {xsd_path}")
    if not documents:
        return Check("C7 XSD round-trip", False, "no documents supplied")

    extractor = ModelExtractor(Path(str(model_dir)), Path(str(tokenizer_dir)))
    rep = run_workflow(documents, extractor, "model", xsd_path)
    rate = rep.xsd_pass_rate
    built = round(rate * len(documents))
    return Check(
        "C7 XSD round-trip",
        rate >= MIN_XSD_PASS_RATE,
        f"{built}/{len(documents)} extractions built a schema-valid message",
        rate,
        MIN_XSD_PASS_RATE,
    )


# --------------------------------------------------------------------------
# The gate itself
# --------------------------------------------------------------------------


def evaluate(
    model: Any,
    tokenizer: Any,
    examples: list[Any],
    device: str = "cpu",
    max_new_tokens: int = 200,
    limit: int | None = None,
) -> AcceptanceReport:
    """Run every acceptance check over held-out documents.

    `model` and `tokenizer` are loosely typed by design: the artifact loads
    through `trust_remote_code`, so its concrete classes live in the Hub copy
    rather than in this tree.

    C8 takes no ground truth, which is the point. C5 is accuracy, and accuracy is
    unavailable at inference time -- so a gate built only on C1-C7 can be passed
    by a pipeline that emits a wrong payment, which is exactly what happened.
    """
    import torch

    from iso20022_lab.model.tokenizer import prompt_for

    report = AcceptanceReport()
    subset = stratified_sample(examples, limit) if limit else examples
    report.docs = len(subset)

    parsed_ok = 0
    clean_keys = 0
    no_dupes = 0
    grounded_ok = 0
    role_clean = 0
    role_examples: list[str] = []
    routing_clean = 0
    routing_unverifiable = 0
    routing_examples: list[str] = []
    fields_expected = 0
    fields_correct = 0
    doc_stp = 0
    value_total = 0

    # C6 is checked on a sample rather than every document: generating twice for
    # all of them doubles the run time for a property that is architectural
    # (greedy decoding, no dropout at eval) rather than content-dependent.
    det_checked = 0
    det_stable = 0
    determinism_sample = subset[: min(3, len(subset))]

    def generate(text: str) -> str:
        ids = tokenizer.encode(prompt_for(text), add_special_tokens=False)
        # pyright: ignore[reportPrivateImportUsage] -- torch re-exports its own
        # public C symbols through a private module, so this check reports
        # `torch.tensor` as non-public. It is the documented API and the flag is
        # the known false positive; the project config disables it, and this
        # keeps the file clean under stricter configs too.
        input_ids: Any = torch.tensor([ids], device=device)  # pyright: ignore[reportPrivateImportUsage]
        with torch.no_grad():
            out = model.generate_greedy(input_ids, max_new_tokens=max_new_tokens)
        return str(tokenizer.decode(out[0, len(ids) :].tolist()))

    for item in subset:
        item_d: dict[str, Any] = dict(item)
        text = str(item_d["text"])
        truth = {str(k): str(v) for k, v in dict(item_d["values"]).items()}

        raw = generate(text)
        payload, _why = parse_strict(raw)

        if payload:
            parsed_ok += 1

        if payload and set(payload) <= CANONICAL_KEYS:
            clean_keys += 1

        if count_duplicate_keys(raw) == 0:
            no_dupes += 1

        # C8: ground-truth-free role and provenance check. This is the one that
        # catches the real failure the other checks let through.
        if payload:
            role_findings = role_consistency(text, payload)
            if not role_findings:
                role_clean += 1
            elif len(role_examples) < 3:
                role_examples.append("; ".join(str(f) for f in role_findings))

        # C9: cross-field routing agreement. The IBAN is checksum-protected and
        # the BIC is not, so the trustworthy field tests the untrustworthy one.
        # Measured need: all 9 silently-wrong fields in the recognition run were
        # the agent BIC, and 8 of 9 passed shape AND country checks.
        if payload:
            findings = cross_check(payload)
            if any(f.blocking for f in findings):
                if len(routing_examples) < 3:
                    failed = [f for f in findings if f.blocking]
                    routing_examples.append(failed[0].detail)
            else:
                routing_clean += 1
            routing_unverifiable += sum(
                1 for f in findings if f.verdict is Verdict.UNVERIFIABLE
            )

        # C4: every emitted value must be traceable to the document or the truth.
        # Normalised, so reformatting is not scored as invention.
        haystack = normalise(text) + "|" + "|".join(normalise(v) for v in truth.values())
        if payload:
            bad = [
                k
                for k, v in payload.items()
                if normalise(v) and normalise(v) not in haystack
            ]
            value_total += len(payload)
            if not bad:
                grounded_ok += 1
            elif len(report.samples) < 3:
                report.samples.append(f"ungrounded {bad} in {raw[:120]}")

        # C5: value accuracy against truth.
        if truth:
            correct_here = sum(1 for k, v in truth.items() if payload.get(k) == v)
            fields_correct += correct_here
            fields_expected += len(truth)
            if correct_here == len(truth) and payload:
                doc_stp += 1

    for item in determinism_sample:
        text = str(dict(item)["text"])
        det_checked += 1
        if generate(text) == generate(text):
            det_stable += 1

    json_rate = parsed_ok / report.docs if report.docs else 0.0
    key_rate = clean_keys / report.docs if report.docs else 0.0
    dupe_rate = no_dupes / report.docs if report.docs else 0.0
    grounded_rate = grounded_ok / report.docs if report.docs else 0.0
    role_rate = role_clean / parsed_ok if parsed_ok else 0.0
    routing_rate = routing_clean / parsed_ok if parsed_ok else 0.0
    accuracy = fields_correct / fields_expected if fields_expected else 0.0
    stp = doc_stp / report.docs if report.docs else 0.0
    det_rate = det_stable / det_checked if det_checked else 0.0

    report.checks = [
        Check(
            "C1 valid JSON object",
            json_rate >= MIN_JSON_PARSE_RATE,
            f"{parsed_ok}/{report.docs} parsed",
            json_rate,
            MIN_JSON_PARSE_RATE,
        ),
        Check(
            "C2 canonical keys only",
            key_rate >= MIN_CLEAN_KEY_RATE,
            f"{clean_keys}/{report.docs} used only known field names",
            key_rate,
            MIN_CLEAN_KEY_RATE,
        ),
        Check(
            "C3 no duplicate keys",
            dupe_rate >= MIN_CLEAN_KEY_RATE,
            f"{no_dupes}/{report.docs} free of repeated keys",
            dupe_rate,
            MIN_CLEAN_KEY_RATE,
        ),
        Check(
            "C4 values grounded",
            grounded_rate >= MIN_GROUNDED_RATE,
            f"{grounded_ok}/{report.docs} wholly traceable ({value_total} values)",
            grounded_rate,
            MIN_GROUNDED_RATE,
        ),
        Check(
            "C8 role consistency",
            role_rate >= MIN_ROLE_CLEAN_RATE,
            f"{role_clean}/{parsed_ok} free of role or provenance errors"
            + (f" -- e.g. {role_examples[0]}" if role_examples else ""),
            role_rate,
            MIN_ROLE_CLEAN_RATE,
        ),
        Check(
            "C9 routing agreement",
            routing_rate >= MIN_ROUTING_CLEAN_RATE,
            f"{routing_clean}/{parsed_ok} free of routing disagreement"
            + (f" ({routing_unverifiable} unverifiable)" if routing_unverifiable else "")
            + (f" -- e.g. {routing_examples[0]}" if routing_examples else ""),
            routing_rate,
            MIN_ROUTING_CLEAN_RATE,
        ),
        Check(
            "C5 field accuracy",
            accuracy >= MIN_FIELD_ACCURACY,
            f"{fields_correct}/{fields_expected} fields correct",
            accuracy,
            MIN_FIELD_ACCURACY,
        ),
        Check(
            "C5b document STP",
            stp >= MIN_STP_RATE,
            f"{doc_stp}/{report.docs} documents fully correct",
            stp,
            MIN_STP_RATE,
        ),
        Check(
            "C6 deterministic",
            det_rate == 1.0,
            f"{det_stable}/{det_checked} repeated identical",
            det_rate,
            1.0,
        ),
    ]
    return report
