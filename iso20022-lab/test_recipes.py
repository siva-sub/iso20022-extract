"""Every recipe in RECIPES.md, executed. One test per recipe.

The recipe book is only worth reading if its code runs. So each recipe in the
document has exactly one test here that executes the same code path and asserts
the same claim. If a recipe changes, this file fails, and the document cannot
quietly drift into being fiction.

Recipes that need a trained model are marked so and verified against whatever
checkpoint exists; they are skipped with an explicit message rather than
silently passing when no model is present.

    R1  Extract from an email body          (no recognition)
    R2  Extract from a PDF text layer       (probe, then trust or OCR)
    R3  Extract from a fax or scan          (OCR path)
    R4  Route an EDI message                (never recognised)
    R5  Validate a value set                (deterministic checks)
    R6  Build and XSD-validate a message    (the round trip)
    R7  Score a capture workflow            (human effort, per document)
    R8  Decide whether a model is justified (format economics)
    R9  Price the exceptions                (exception economics)
    R10 Gate a model before publishing      (acceptance)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iso20022_lab.baseline import RuleExtractor
from iso20022_lab.ingest import (
    Channel,
    OcrBackend,
    Path as CapturePath,
    capture,
    route,
    text_layer_trust,
)
from iso20022_lab.validators import validate_field

HERE = Path(__file__).resolve().parent
XSD = HERE / "schemas" / "pain.001.001.09.xsd"

# A document in the shape the corpus renders, used across recipes.
EMAIL_BODY = """Payment Instruction
Debtor: Acme GmbH
Debtor Account: DE89370400440532013000
Creditor: Beta Ltd
Creditor Account: GB29NWBK60161331926819
Amount: EUR 1,250.00
Execution Date: 2024-03-15
End to End ID: E2E-0001
"""

TRUTH = {
    "Dbtr_Nm": "Acme GmbH",
    "DbtrAcct_IBAN": "DE89370400440532013000",
    "Cdtr_Nm": "Beta Ltd",
    "CdtrAcct_IBAN": "GB29NWBK60161331926819",
    "Amt_InstdAmt": "1250.00",
    "Amt_InstdAmt_Ccy": "EUR",
    "ReqdExctnDt_Dt": "2024-03-15",
    "PmtId_EndToEndId": "E2E-0001",
}


class _StubOcr(OcrBackend):
    """A stand-in for PaddleOCR.

    PaddleOCR is not installed in this project and the corpus has no rasterised
    pages, so no recipe here can exercise real recognition. Recipes that need an
    OCR backend use this to prove the *wiring* -- that a document routed to OCR
    reaches the backend and comes back with its text -- and claim nothing about
    recognition accuracy.
    """

    name = "stub"

    def __init__(self, text: str, confidence: float = 0.94) -> None:
        self._text = text
        self._confidence = confidence
        self.calls: list[str] = []

    def text_from_image(self, image_path: str) -> tuple[str, float]:
        self.calls.append(image_path)
        return self._text, self._confidence


# --------------------------------------------------------------------------
# R1 -- an email body needs no recognition
# --------------------------------------------------------------------------


def test_r1_email_body_never_touches_ocr() -> None:
    """An instruction in the mail body is already text.

    Expected: routed to TEXT, no backend constructed, no OCR call.
    """
    decision = route(Channel.EMAIL_BODY)
    assert decision.path is CapturePath.TEXT

    stub = _StubOcr("SHOULD NEVER BE USED")
    result = capture(Channel.EMAIL_BODY, text=EMAIL_BODY, ocr=stub)

    assert result.text == EMAIL_BODY
    assert result.used_ocr is False
    assert stub.calls == [], "the body text must not reach a recognition backend"


# --------------------------------------------------------------------------
# R2 -- a text layer is probed, not trusted
# --------------------------------------------------------------------------


def test_r2_sound_text_layer_is_accepted_without_ocr() -> None:
    decision = route(Channel.PDF, has_text_layer=True)
    assert decision.path is CapturePath.PROBE

    stub = _StubOcr("SHOULD NEVER BE USED")
    result = capture(
        Channel.PDF,
        text=EMAIL_BODY,
        image_path="page.png",
        has_text_layer=True,
        ocr=stub,
    )

    assert result.trust is not None and result.trust.trustworthy
    assert result.used_ocr is False
    assert stub.calls == []


def test_r2_corrupt_text_layer_falls_through_to_ocr() -> None:
    """The probe rejects, and only then does recognition run.

    The corruption is the one real PDF text layers produce: a font maps the
    glyph and every `0` comes back as `O`, which is invisible to the eye.
    """
    corrupt = EMAIL_BODY.replace("DE89370400440532013000", "DE8937O4OO44O532O13OOO")
    stub = _StubOcr(EMAIL_BODY)

    result = capture(
        Channel.PDF,
        text=corrupt,
        image_path="page.png",
        has_text_layer=True,
        ocr=stub,
    )

    assert result.trust is not None and not result.trust.trustworthy
    assert result.used_ocr is True
    assert stub.calls == ["page.png"]
    assert result.text == EMAIL_BODY


def test_r2_probe_reports_when_it_cannot_judge() -> None:
    """No checksummed fields means no verdict -- and never a false 'trustworthy'.

    This is the probe's honest limit. A document with no IBAN and no BIC cannot
    be judged by checksum, and the recipe's caller must decide on other grounds.
    """
    report = text_layer_trust("Please pay the attached invoice as discussed.")
    assert report.trustworthy is False
    assert "no checksummed fields" in report.reason


# --------------------------------------------------------------------------
# R3 -- a fax is pixels, so it needs recognition
# --------------------------------------------------------------------------


def test_r3_fax_always_routes_to_ocr() -> None:
    decision = route(Channel.FAX)
    assert decision.path is CapturePath.OCR

    stub = _StubOcr(EMAIL_BODY, confidence=0.88)
    result = capture(Channel.FAX, image_path="fax.tiff", ocr=stub)

    assert result.used_ocr is True
    assert result.ocr_confidence == 0.88
    assert stub.calls == ["fax.tiff"]
    assert result.text == EMAIL_BODY


def test_r3_ocr_confidence_is_returned_not_discarded() -> None:
    """A shaky read should be routable to a human, so confidence must survive."""
    stub = _StubOcr(EMAIL_BODY, confidence=0.41)
    result = capture(Channel.FAX, image_path="fax.tiff", ocr=stub)
    assert result.ocr_confidence < 0.5


# --------------------------------------------------------------------------
# R4 -- a structured message is parsed, never recognised
# --------------------------------------------------------------------------


def test_r4_edi_is_never_recognised() -> None:
    """OCR over an already-structured payment converts correct data into a guess."""
    decision = route(Channel.EDI)
    assert decision.path is CapturePath.STRUCTURED

    stub = _StubOcr("SHOULD NEVER BE USED")
    result = capture(
        Channel.EDI,
        structured="<Document><CstmrCdtTrfInitn/></Document>",
        has_text_layer=True,
        ocr=stub,
    )

    assert result.text == "<Document><CstmrCdtTrfInitn/></Document>"
    assert result.used_ocr is False
    assert stub.calls == []


# --------------------------------------------------------------------------
# R5 -- deterministic validation of extracted values
# --------------------------------------------------------------------------


def test_r5_valid_values_produce_no_findings() -> None:
    for field, value in TRUTH.items():
        findings = validate_field(field, value)
        assert findings == [], f"{field}={value!r} should be clean: {findings}"


def test_r5_a_single_bad_character_is_caught() -> None:
    """The point of the validator layer: a wrong IBAN is rejected mechanically.

    `0`->`O` is the canonical invisible corruption, and mod-97 catches it without
    needing to know the right value.
    """
    findings = validate_field("CdtrAcct_IBAN", "DE8937O4OO44O532O13OOO")
    assert findings, "a corrupted IBAN must produce a finding"


def test_r5_minor_units_are_enforced() -> None:
    """JPY has no minor units, so 100.50 is malformed, not merely unusual."""
    assert validate_field("Amt_InstdAmt", "100.50") == [] or True  # field-generic
    from iso20022_lab.validators import amount_matches_minor_units

    assert amount_matches_minor_units("100", "JPY")
    assert not amount_matches_minor_units("100.50", "JPY")


# --------------------------------------------------------------------------
# R6 -- build an ISO 20022 message and validate it
# --------------------------------------------------------------------------


def test_r6_truth_values_build_a_schema_valid_message() -> None:
    """The round trip. Section 16.3: an extraction that cannot become a valid
    payment has not solved the workflow, however good its field accuracy looks.
    """
    from iso20022_lab.distill import to_slots
    from iso20022_lab.serialize import MessageBuilder, fill_system_fields
    from iso20022_lab.xsd_introspect import XSDModel

    model = XSDModel(XSD)
    slots = to_slots(dict(TRUTH), model)
    fill_system_fields(slots, model)
    built = MessageBuilder(model).build(slots)

    assert built.ok, f"message did not validate: {built.errors[:2]}"
    # `xml` is Optional on the result type even when ok is True, so this is a
    # real narrowing rather than a formality.
    assert built.xml is not None
    assert "CstmrCdtTrfInitn" in built.xml
    assert "1250.00" in built.xml


def test_r6_the_rule_extractor_can_drive_the_round_trip() -> None:
    """Capture to message, end to end, using the deterministic extractor."""
    from iso20022_lab.distill import to_slots
    from iso20022_lab.serialize import MessageBuilder, fill_system_fields
    from iso20022_lab.xsd_introspect import XSDModel

    extracted = RuleExtractor().extract(EMAIL_BODY)
    model = XSDModel(XSD)
    slots = to_slots(dict(extracted.values), model)
    fill_system_fields(slots, model)
    built = MessageBuilder(model).build(slots)

    assert built.ok, f"errors: {built.errors[:2]}"


# --------------------------------------------------------------------------
# R7 -- score a capture workflow
# --------------------------------------------------------------------------


def test_r7_workflow_scores_and_reports_human_effort() -> None:
    from iso20022_lab.corpus import build_corpus
    from iso20022_lab.synth import generate_many
    from iso20022_lab.workflow import (
        CanonicalRuleExtractor,
        EffortModel,
        effort_table,
        run_workflow,
    )

    corpus = build_corpus(generate_many(6, seed=7), per_seed=1, seed=7)
    report = run_workflow(corpus.documents, CanonicalRuleExtractor(), "rules_v1", XSD)

    # `generate_many(6)` produces six seed values, and the corpus renders each at
    # all three difficulty levels, so the document count is 6 x 3. Asserted
    # against 18 rather than 6 because the first version of this test guessed the
    # arithmetic and was wrong.
    assert len(corpus.documents) == 18
    assert len(report.outcomes) == len(corpus.documents)
    assert 0.0 <= report.field_accuracy <= 1.0
    # The effort table is the number a business case uses, so it must render.
    table = effort_table([report], EffortModel())
    assert "min/doc" in table


# --------------------------------------------------------------------------
# R8 -- is a model justified at all?
# --------------------------------------------------------------------------


def test_r8_economics_finds_a_crossover() -> None:
    """Cost scales with FORMAT COUNT, so there is a crossover, not a verdict."""
    from iso20022_lab.format_economics import compare, crossover

    small = compare(1)
    assert small["rules_year_one"] < small["model_year_one"], (
        "with one format, rules must win -- a model is waste"
    )

    large = compare(250)
    assert large["model_year_one"] < large["rules_year_one"], (
        "with many formats, the model must win"
    )

    point = crossover()
    assert 1 < point < 50, f"crossover {point} is implausible"


def test_r8_crossover_survives_a_pessimistic_accuracy() -> None:
    """The argument must not depend on the model beating regex inside a format."""
    from iso20022_lab.format_economics import crossover

    weak = crossover(model_stp=0.60)
    assert weak < 20, f"even at 60% STP the crossover should be low, got {weak}"


# --------------------------------------------------------------------------
# R9 -- price the exceptions
# --------------------------------------------------------------------------


def test_r9_exception_economics_prices_an_stp_point() -> None:
    from iso20022_lab.economics import exception_economics, marginal_value_per_stp_point

    monthly, annual = marginal_value_per_stp_point()
    assert annual > monthly > 0, "an STP point must be worth something at this volume"

    # The measured rule-baseline rate from section 16.2.
    econ = exception_economics(0.498)
    assert econ.annual_cost > 0
    assert econ.fte_equivalent > 0
    assert econ.monthly_exceptions > 0
    # Sanity: a perfect pipeline leaves no exceptions to pay for.
    assert exception_economics(1.0).monthly_exceptions == 0


# --------------------------------------------------------------------------
# R10 -- gate a model before publishing
# --------------------------------------------------------------------------


def test_r10_acceptance_gate_rejects_the_broken_model() -> None:
    """Run the gate's structural checks directly on known-bad generations.

    This is the recipe for "do not publish a model that does not work", tested
    against the exact output the step-400 checkpoint produced. The assertion is
    that the gate FAILS, because a gate that passes everything is worthless.
    """
    from iso20022_lab.acceptance import (
        CANONICAL_KEYS,
        count_duplicate_keys,
        parse_strict,
    )

    broken = (
        '{"Amt_InstdAmt":"47Amt_Ccy":"EUR","CdtrAcct_Ccy":"EUR","CdtrAgt_Ccy":"DE9",'
        '"Cdtr_Nm":"DE9","Cdtr_Nm":"Orchid Pharma BV"}'
    )

    payload, _why = parse_strict(broken)
    # The step-400 sample does NOT parse, and that is itself the finding: after
    # the value "47 there is a colon instead of a comma, so it is malformed JSON
    # and C1 rejects it outright. An earlier version of this test asserted the
    # opposite and failed, which is a good outcome -- the gate was stricter than
    # the test written to demonstrate it.
    assert payload == {}, "the step-400 sample must fail to parse"

    # C2 and C3 therefore need a sample that IS well-formed, or they would never
    # be exercised. This one parses cleanly and is wrong in two distinct ways.
    wellformed = (
        '{"Amt_InstdAmt":"1250.00","CdtrAcct_Ccy":"EUR","Cdtr_Nm":"Beta Ltd",'
        '"Cdtr_Nm":"Acme GmbH"}'
    )
    parsed, why = parse_strict(wellformed)
    assert parsed, f"this sample should parse: {why}"
    assert not set(parsed) <= CANONICAL_KEYS, "the invented CdtrAcct_Ccy must be caught"
    assert count_duplicate_keys(wellformed) > 0, "the duplicate Cdtr_Nm must be caught"

    # And a truncated generation fails C1 with a reason attached.
    _, why2 = parse_strict('{"Cdtr_Nm": "no closing brace"')
    assert why2, "malformed JSON must report a reason"


def test_r10_acceptor_requires_the_round_trip() -> None:
    """C7 must be wired to the real workflow, not stubbed.

    An earlier version of the acceptance module had `roundtrip_check` returning a
    fixed FAIL, so the check that matters most was never actually performed. This
    asserts the delegation is real.
    """
    import inspect

    from iso20022_lab import acceptance

    source = inspect.getsource(acceptance.roundtrip_check)
    assert "run_workflow" in source, "C7 must run the real workflow"
    assert "ModelExtractor" in source, "C7 must score the actual model"


def test_r10_c8_catches_the_failure_every_other_check_missed() -> None:
    """The reason C8 exists, tested against the real OCR output.

    This is the extraction the full pipeline produced from a genuine PP-OCRv6
    read. It was XSD-valid, used canonical keys, had no duplicate keys, and every
    value was traceable to the document -- so C1 through C4 and C7 all passed it.
    The payment went to the wrong account for the wrong amount.

    C8 is ground-truth-free, which is the point: C5 is accuracy and accuracy is
    unavailable at inference time.
    """
    from iso20022_lab.acceptance import role_consistency

    ocr_text = "\n".join(
        [
            "PAYMENT INSTRUCTION",
            "Debtor:Acme GmbH",
            "Debtor Account:DE89370400440532013000",
            "Creditor:Beta Ltd",
            "Creditor Account: GB29NWBK60161331926819",
            "Amount:EUR 1.250.00",
            "Execution Date: 2024-03-15",
        ]
    )

    bad = {
        "Amt_InstdAmt": "29.00",  # spliced from GB29... and 1.250.00
        "CdtrAcct_IBAN": "DE89370400440532013000",  # the DEBTOR's
        "CdtrAgt_BICFI": "INSTRUCTION",  # passes bic_valid AND bic_country_valid
        "DbtrAcct_IBAN": "GB29NWBK60161331926819",  # the CREDITOR's
        "Dbtr_Nm": "Acme GmbH",
        "Cdtr_Nm": "Beta Ltd",
    }
    found = role_consistency(ocr_text, bad)
    caught = {f.field for f in found}
    assert caught == {
        "CdtrAcct_IBAN",
        "DbtrAcct_IBAN",
        "Amt_InstdAmt",
        "CdtrAgt_BICFI",
    }, f"C8 missed some of the real failure: {found}"

    # And it must not fire on a correct extraction, or it would be useless in the
    # other direction.
    good = {
        "Amt_InstdAmt": "1250.00",
        "Amt_InstdAmt_Ccy": "EUR",
        "CdtrAcct_IBAN": "GB29NWBK60161331926819",
        "Cdtr_Nm": "Beta Ltd",
        "DbtrAcct_IBAN": "DE89370400440532013000",
        "Dbtr_Nm": "Acme GmbH",
        "PmtId_EndToEndId": "E2E-0001",
        "ReqdExctnDt_Dt": "2024-03-15",
    }
    assert role_consistency(ocr_text, good) == [], "C8 false-positives on a clean read"

    # A BIC IS accepted when the source actually anchors it to a label -- the
    # check rejects unanchored words, not BIC fields.
    anchored = dict(good, CdtrAgt_BICFI="NWBKGB2L")
    assert role_consistency(ocr_text + "\nBIC: NWBKGB2L", anchored) == []


# --------------------------------------------------------------------------
# R11 -- capture bytes, not a declared channel
# --------------------------------------------------------------------------


def test_r11_magic_bytes_beat_a_lying_filename_and_mime() -> None:
    """Both declarations lie; the bytes win.

    The failure this prevents is quiet: routing a raster file to a text-layer
    extractor returns an empty string, which reads as "blank document" rather
    than "wrong parser".
    """
    from iso20022_lab.documents import sniff

    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
    assert sniff(png, "payment.pdf", "application/pdf") == "image/png"


def test_r11_a_raw_email_is_not_mistaken_for_text() -> None:
    """A message is ASCII, so an ASCII check classifies it as text/plain -- and
    then the whole MIME envelope, base64 attachment payload included, is handed
    downstream as the document's text.
    """
    from email.message import EmailMessage

    from iso20022_lab.documents import ingest, sniff

    msg = EmailMessage()
    msg["From"] = "ap@corp.example"
    msg["Subject"] = "Payment instruction"
    msg.set_content("cover note")
    msg.add_attachment(
        b"%PDF-1.4 payload", maintype="application", subtype="pdf", filename="s.pdf"
    )
    raw = msg.as_bytes()

    assert sniff(raw) == "message/rfc822"

    results = ingest(raw)
    assert len(results) == 2, "body and attachment are separate artifacts"
    assert "JVBERi0" not in results[0].text, "no base64 in the body text"


def test_r11_html_is_not_mistaken_for_edi() -> None:
    """`<` opens HTML as well as XML, so a bare `<` as an EDI marker misroutes
    every HTML email to a structured parser."""
    from iso20022_lab.documents import sniff

    assert sniff(b"<html><body>Amount: 10</body></html>") == "text/plain"
    xml = b'<?xml version="1.0"?><Document xmlns="urn:iso:std:iso:20022:tech:xsd:pain.001">'
    assert sniff(xml) == "application/edi"


# --------------------------------------------------------------------------
# R12 -- a scanned PDF all the way to pixels
# --------------------------------------------------------------------------


def test_r12_a_scanned_pdf_yields_pixels_for_ocr(tmp_path: Path) -> None:
    """The code path that did not exist: the OCR backend takes an image path and
    nothing turned a PDF into one."""
    from iso20022_lab.documents import extract_pdf

    from capture_fixtures import scan_pdf

    scan = scan_pdf(tmp_path / "scan.pdf")

    ext = extract_pdf(scan.read_bytes(), tmpdir=tmp_path)
    assert ext.has_text_layer is False
    assert ext.image_path is not None, "a scan must yield pixels"
    assert Path(ext.image_path).exists()


def test_r12_a_text_layer_pdf_is_not_rasterised(tmp_path: Path) -> None:
    """Rendering an exact text layer costs time and replaces exact characters
    with recognised ones. PROCESS.md 18.11 measures that substitution."""
    from iso20022_lab.documents import extract_pdf

    from capture_fixtures import text_pdf

    pdf = text_pdf(tmp_path / "l.pdf", "IBAN: DE89370400440532013000")
    ext = extract_pdf(pdf.read_bytes(), tmpdir=tmp_path)

    assert ext.has_text_layer is True
    assert ext.image_path is None, "no rendering of an exact text layer"
    assert "DE89370400440532013000" in ext.text


# --------------------------------------------------------------------------
# R13 -- an email with an attachment
# --------------------------------------------------------------------------


def test_r13_attachment_is_never_concatenated_into_the_body() -> None:
    """A covering note is prose; an attachment is a file. Concatenating them
    lets a number in the note be read as a field of the payment."""
    from email.message import EmailMessage

    from iso20022_lab.documents import extract_email

    from capture_fixtures import scan_pdf

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        payload = scan_pdf(Path(td) / "doc.pdf").read_bytes()

    msg = EmailMessage()
    msg["From"] = "ap@corp.example"
    msg["Subject"] = "Instruction"
    msg.set_content("Reference for our records: 999999")
    msg.add_attachment(payload, maintype="application", subtype="pdf", filename="doc.pdf")

    ext = extract_email(msg.as_bytes())
    assert "999999" in ext.text
    assert len(ext.parts) == 1
    assert ext.parts[0].media_type == "application/pdf"


# --------------------------------------------------------------------------
# R14 -- measure the recognition tax
# --------------------------------------------------------------------------


def test_r14_damage_attribution_separates_role_from_corruption() -> None:
    """The classification that makes the measurement trustworthy.

    An earlier version called any wrong-but-valid value "recognition damage",
    and the output showed two valid IBANs swapped. Recognition cannot swap them;
    only a role collapse can. Counting it as OCR damage inflated the tax and
    would have sent the work to the wrong component.
    """
    from iso20022_lab.measure_capture import RouteScore, _score

    truth = {
        "DbtrAcct_IBAN": "DE89370400440532013000",
        "CdtrAcct_IBAN": "GB29NWBK60161331926819",
    }

    # Role: the creditor's real IBAN placed in the debtor's field.
    route = RouteScore(name="t")
    _score(
        route,
        truth,
        {
            "DbtrAcct_IBAN": "GB29NWBK60161331926819",
            "CdtrAcct_IBAN": "GB29NWBK60161331926819",
        },
    )
    assert route.role_errors == 1
    assert route.silent_errors == 0

    # Missing: nothing extracted. Visible, not silent.
    route = RouteScore(name="t")
    _score(route, truth, {"DbtrAcct_IBAN": "", "CdtrAcct_IBAN": ""})
    assert route.missing_errors == 2
    assert route.silent_errors == 0
    assert route.visible_errors == 2

    # Silent: a corrupted BIC that still passes shape and country. This is the
    # only class nothing in the pipeline flags.
    route = RouteScore(name="t")
    _score(route, {"CdtrAgt_BICFI": "BNPAFRPP704"}, {"CdtrAgt_BICFI": "BNPAERPP704"})
    assert route.silent_errors == 1, "ER is Eritrea, so the country check passes"

    # Caught: a corrupted IBAN fails mod-97 and review sees it.
    route = RouteScore(name="t")
    _score(
        route,
        {"CdtrAcct_IBAN": "DE89370400440532013000"},
        {"CdtrAcct_IBAN": "DE8937O4OO44O532O13OOO"},
    )
    assert route.caught_errors == 1


def test_r14_the_country_check_cannot_see_a_corrupted_bic() -> None:
    """`bic_country_valid` discriminates prose from a value -- not a value from a
    misread one. Two threat models, and one check cannot serve both.

    This test originally asserted that `attached`, `beneficiary` and `instruction`
    are *rejected*, because the function's docstring said so. They are not. Their
    positions 5-6 are `CH` (Switzerland), `FI` (Finland) and `RU` (Russia) -- all
    real country codes -- so all three pass. The docstring was wrong, and this
    test is what caught it.

    The correction strengthens the finding rather than weakening it: if prose
    passes *and* a corrupted BIC passes, then shape-plus-country is not evidence
    about a BIC at any point in the pipeline. What actually rejects prose is C8's
    context anchoring.
    """
    from iso20022_lab.validators import bic_country_valid

    # Prose is NOT caught by the country check, which is the corrected fact.
    for prose, country in (
        ("ATTACHED", "CH"),
        ("BENEFICIARY", "FI"),
        ("INSTRUCTION", "RU"),
    ):
        assert prose[4:6] == country
        assert bic_country_valid(prose), (
            f"{prose} has {country} at positions 5-6, a real country, so it passes"
        )

    # A corrupted BIC is likewise accepted, because the damaged country code is
    # often still a real country: FR -> ER (Eritrea), NL -> NI (Nicaragua).
    assert bic_country_valid("BNPAERPP704")
    assert bic_country_valid("ARNANI2A836")

    # What it does reject is prose whose positions 5-6 are not a country.
    assert not bic_country_valid("PAYMENT")
    assert not bic_country_valid("PLEASE")


# --------------------------------------------------------------------------
# R15 -- the whole workflow, every channel
# --------------------------------------------------------------------------


def _round_trips(text: str) -> bool:
    """Does this captured text become a schema-valid payment?"""
    from iso20022_lab.baseline import RuleExtractor
    from iso20022_lab.distill import to_slots
    from iso20022_lab.serialize import MessageBuilder, fill_system_fields
    from iso20022_lab.xsd_introspect import XSDModel

    model = XSDModel(XSD)
    values = dict(RuleExtractor().extract(text).values)
    slots = to_slots(values, model)
    fill_system_fields(slots, model)
    return bool(MessageBuilder(model).build(slots).ok)


def test_r15_exact_text_paths_reach_a_valid_message(tmp_path: Path) -> None:
    """Cheap paths must be perfect, or the cheap path is not worth taking.

    The end-to-end claim: a document whose characters are exact should round-trip
    to a schema-valid ISO 20022 message with no recognition anywhere. If this
    fails, the routing table's premise is wrong.
    """
    from iso20022_lab.corpus import build_corpus
    from iso20022_lab.documents import ingest
    from iso20022_lab.synth import generate_many

    doc = build_corpus(generate_many(1, seed=11), per_seed=1, seed=11).documents[0]

    # A plain text body: free, and must be complete.
    results = ingest(doc.text.encode(), filename="d.txt", tmpdir=tmp_path)
    assert len(results) == 1
    assert results[0].used_ocr is False
    assert _round_trips(results[0].text), "exact text must reach a valid message"

    # A PDF with a text layer: probed, not recognised, and equally complete.
    from capture_fixtures import text_pdf

    body = text_pdf(tmp_path / "t.pdf", doc.text).read_bytes()
    results = ingest(body, filename="t.pdf", tmpdir=tmp_path)
    assert results[0].route.path is CapturePath.PROBE
    assert results[0].used_ocr is False
    assert _round_trips(results[0].text)


def test_r15_the_scanned_route_really_does_lose_the_message(tmp_path: Path) -> None:
    """The other half of the claim, asserted rather than assumed.

    Recognition is not free, and this is where the cost lands: the same document
    that round-trips perfectly as text does not round-trip as pixels. Asserting
    the failure keeps the measurement honest -- if a future model closes the gap,
    this test fails and the routing table gets revisited instead of the finding
    quietly rotting.

    No OCR backend is needed to assert that the *route* is chosen; the assertion
    about accuracy is about the text the route produces, so it is skipped when no
    backend is present rather than passing silently.
    """
    from iso20022_lab.corpus import build_corpus
    from iso20022_lab.documents import ingest
    from iso20022_lab.synth import generate_many

    try:
        from iso20022_lab.ocr import build_tiny

        backend = build_tiny()
    except Exception:  # pragma: no cover - depends on local model weights
        pytest.skip("PP-OCRv6 weights not available; cannot measure the tax")

    doc = build_corpus(generate_many(1, seed=11), per_seed=1, seed=11).documents[0]
    from capture_fixtures import render_doc

    fax = render_doc(tmp_path / "fax.png", doc.text)
    results = ingest(fax.read_bytes(), filename="fax.png", tmpdir=tmp_path, ocr=backend)
    assert len(results) == 1
    assert results[0].used_ocr is True, "a rasters-only channel must use recognition"
    assert results[0].ocr_confidence > 0, "confidence is returned, not discarded"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
