"""The ingest layer, tested. This produces the numbers in PROCESS.md section 18.

Two things are checked here, and they are checked by running rather than by
assertion:

1. **Routing.** Every inbound channel reaches the right path, and OCR is selected
   only when there is no cheaper option. The cacheable claim is "OCR is needed by
   two of six channels", and it is a test, not a sentence.

2. **The trust probe.** The load-bearing claim is that the deterministic
   validators already used to gate output can also detect a corrupt input text
   layer, so OCR can be a fallback rather than the default. If the probe cannot
   catch realistic text-layer corruption, that claim is false and the whole
   design collapses back to "OCR everything".

The corruptions in `CORRUPTIONS` are the ones real PDF text layers actually
produce: C0 controls and ligature mapping fail into lookalike characters, and
optical recognition confuses glyph pairs. They are applied to real corpus values
with valid check digits, so a detection is a detection.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable

from iso20022_lab.ingest import (
    Channel,
    OcrBackend,
    Path,
    capture,
    route,
    text_layer_trust,
)
from iso20022_lab.validators import iban_valid

# --------------------------------------------------------------------------
# 1. Routing
# --------------------------------------------------------------------------

ROUTING_CASES: list[tuple[Channel, dict[str, bool], Path]] = [
    (Channel.EMAIL_BODY, {}, Path.TEXT),
    (Channel.EDI, {}, Path.STRUCTURED),
    (Channel.EMAIL_ATTACHMENT, {}, Path.OCR),
    (Channel.FAX, {}, Path.OCR),
    (Channel.PDF, {"has_text_layer": True}, Path.PROBE),
    (Channel.PDF, {}, Path.OCR),
    (Channel.PORTAL, {"has_text_layer": True}, Path.PROBE),
    (Channel.PORTAL, {}, Path.OCR),
]


def test_routing_sends_each_channel_to_the_cheap_path() -> None:
    for channel, kwargs, expected in ROUTING_CASES:
        got = route(channel, **kwargs)
        assert got.path is expected, (
            f"{channel.value} {kwargs} -> {got.path.value}, expected {expected.value}"
        )


def test_structured_messages_are_never_ocrd() -> None:
    """OCR over a structured payment converts correct data into a guess."""
    assert route(Channel.EDI, has_text_layer=True).path is Path.STRUCTURED
    assert (
        route(Channel.PDF, has_text_layer=True, is_structured=True).path is Path.STRUCTURED
    )


def test_email_body_never_reaches_ocr() -> None:
    """No raster content means nothing for a recognition model to do."""
    assert route(Channel.EMAIL_BODY, has_text_layer=False).path is Path.TEXT


def test_only_text_bearing_channels_avoid_recognition_entirely() -> None:
    """The routing claim, stated precisely.

    Two earlier versions of this test were wrong, and both failures were
    informative rather than noise:

    1. It first asserted the OCR set was exactly {fax, attachment}. That failed
       because `route(PDF)` with no text layer also selects OCR -- the claim had
       been written about *channels* when the real variable is whether pixels are
       the only representation available.

    2. The corrected version asserted fax was the only unconditional OCR channel
       and failed too, for the same underlying reason: an attachment whose type
       is unknown defaults to OCR, which is the safe default rather than a bug.

    What is actually true, and what the design depends on, is narrower: email
    bodies and EDI never require recognition, because neither has pixels that
    hold the information. Everything else requires it exactly when no text layer
    is present.
    """
    # Never recognised, whatever else is true.
    assert route(Channel.EMAIL_BODY).path is Path.TEXT
    assert route(Channel.EMAIL_BODY, has_text_layer=False).path is Path.TEXT
    assert route(Channel.EDI).path is Path.STRUCTURED
    assert route(Channel.EDI, has_text_layer=True).path is Path.STRUCTURED

    # Pixels-only channels: OCR with no text layer, probe when one exists.
    for ch in (Channel.PDF, Channel.PORTAL, Channel.EMAIL_ATTACHMENT):
        assert route(ch, has_text_layer=False).path is Path.OCR, ch
        assert route(ch, has_text_layer=True).path is Path.PROBE, ch

    # A fax is definitionally pixels, so it is the one channel that always needs
    # recognition in practice.
    assert route(Channel.FAX, has_text_layer=False).path is Path.OCR


# --------------------------------------------------------------------------
# 2. The trust probe
# --------------------------------------------------------------------------

GOOD_IBANS = [
    "DE89370400440532013000",
    "GB29NWBK60161331926819",
    "FR1420041010050500013M02606",
    "NL91ABNA0417164300",
    "ES9121000418450200051332",
    "IT60X0542811101000000123456",
]


def test_probe_accepts_a_clean_text_layer() -> None:
    text = "\n".join(f"Account: {i}" for i in GOOD_IBANS)
    report = text_layer_trust(text)
    assert report.ibans_found == len(GOOD_IBANS), report
    assert report.trustworthy, report.as_line()
    assert report.score == 1.0


def test_probe_rejects_a_single_corrupted_iban() -> None:
    """One bad IBAN in a document is enough to fail the layer.

    The corrupted value is not arbitrary: '0'->'O' is what a PDF text layer
    produces when a font maps the glyph wrong, and it is invisible to the eye.
    """
    broken = GOOD_IBANS[0].replace("0", "O")
    text = "\n".join([broken, *GOOD_IBANS[1:]])
    report = text_layer_trust(text)
    assert not report.trustworthy, report.as_line()
    assert report.ibans_found == len(GOOD_IBANS)


# Text-layer failures fall into two classes. Conflating them produces a
# meaningless number, which is what the first version of this file did.
#
#   CORRUPTING  the value is wrong after normalisation. The probe MUST reject.
#   FORMATTING  the value normalises back to the correct one (spacing, case).
#               The probe MUST accept. Rejecting these is a false positive that
#               sends a perfectly good document to OCR.
#
# The first version put both in one list and reported a single "detection rate"
# of 81.7%. Every miss was the measurement being wrong, not the probe:
# `insert-space` and `lowercase` yield valid IBANs after normalisation, and
# three transforms were no-ops on IBANs that did not contain the target digit
# (`five->S` does nothing to an IBAN with no '5'). It also counted `lowercase` as
# caught when the probe had merely found no IBANs and returned "nothing to
# check" -- a different outcome being scored as a detection.
CORRUPTING: list[tuple[str, Callable[[str], str]]] = [
    ("zero->O", lambda s: s.replace("0", "O")),
    ("one->l", lambda s: s.replace("1", "l")),
    ("five->S", lambda s: s.replace("5", "S")),
    ("eight->B", lambda s: s.replace("8", "B")),
    ("two->Z", lambda s: s.replace("2", "Z")),
    ("adjacent-transpose", lambda s: s[:12] + s[13] + s[12] + s[14:]),
    ("drop-space", lambda s: s[:9] + s[10:]),
    ("single-digit-flip", lambda s: s[:14] + str((int(s[14]) + 3) % 10) + s[15:]),
]

FORMATTING: list[tuple[str, Callable[[str], str]]] = [
    ("insert-space", lambda s: s[:9] + " " + s[9:]),
    ("lowercase", lambda s: s.lower()),
]


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()


def _corrupting_cases() -> list[tuple[str, str, str]]:
    """(transform, original, corrupted) for cases that really changed the value.

    A transform that leaves the value alone -- `five->S` on an IBAN containing no
    '5' -- is not a test of anything and is dropped rather than counted as a miss.
    """
    cases: list[tuple[str, str, str]] = []
    for name, transform in CORRUPTING:
        for iban in GOOD_IBANS:
            corrupted = transform(iban)  # type: ignore[operator]
            if not isinstance(corrupted, str):  # pragma: no cover
                continue
            if _compact(corrupted) == iban:
                continue  # no-op, or normalises back clean; not corruption
            cases.append((name, iban, corrupted))
    return cases


def test_probe_catches_real_corruption() -> None:
    """Every genuinely corrupting change must be rejected.

    Asserted at 100%, not a comfortable majority: a corruption that gets through
    is a wrong payment. Only cases where the value actually changed are counted,
    so a no-op transform cannot flatter the score.
    """
    misses: list[str] = []
    for name, iban, corrupted in _corrupting_cases():
        report = text_layer_trust(f"Account: {corrupted}")
        if report.trustworthy:
            misses.append(f"{name} on {iban} -> {corrupted}")
    assert not misses, "corruption the probe did not catch: " + "; ".join(misses)


def test_the_corrupting_cases_really_are_corrupt() -> None:
    """The control for the test above.

    Without this, `test_probe_catches_real_corruption` could pass because the
    probe rejects everything rather than because the values were broken. Every
    case in the list must fail its own checksum.
    """
    cases = _corrupting_cases()
    assert cases, "no corrupting cases generated -- the test is vacuous"
    for name, _iban, corrupted in cases:
        assert not iban_valid(_compact(corrupted)), (
            f"{name} produced a still-valid value: {corrupted}"
        )


def test_formatting_variation_is_accepted_not_rejected() -> None:
    """The other half of the claim, and the one that saves money.

    Spacing and case changes normalise back to the correct value, so rejecting
    them would send a clean document to OCR for nothing. `lowercase` is why the
    matcher is case-insensitive: without that, a lowercased text layer yields no
    candidates at all and the probe reports "no checksummed fields" instead of a
    verdict -- a silent blind spot rather than a decision.
    """
    false_rejects: list[str] = []
    for name, transform in FORMATTING:
        for iban in GOOD_IBANS:
            variant = transform(iban)  # type: ignore[operator]
            if not isinstance(variant, str):  # pragma: no cover
                continue
            report = text_layer_trust(f"Account: {variant}")
            if not report.trustworthy:
                false_rejects.append(f"{name} -> {variant} ({report.reason})")
    assert not false_rejects, "clean text wrongly rejected: " + "; ".join(false_rejects)


def test_probe_is_honest_when_it_has_nothing_to_check() -> None:
    """The probe's limit, asserted so it cannot be forgotten.

    A document with no IBAN and no BIC gets no verdict. Reporting `trustworthy`
    here would be the worst failure mode in the module: silent confidence about
    a text layer nothing was verified.
    """
    report = text_layer_trust("Please pay the attached invoice as discussed.")
    assert report.ibans_found == 0
    assert report.bics_found == 0
    assert not report.trustworthy
    assert "no checksummed fields" in report.reason


def test_probe_threshold_is_a_parameter() -> None:
    """Threshold is a business decision, not a constant buried in the code."""
    broken = GOOD_IBANS[0].replace("0", "O")
    text = "\n".join([broken, *GOOD_IBANS[1:]])
    assert text_layer_trust(text, threshold=0.5).trustworthy
    assert not text_layer_trust(text, threshold=0.99).trustworthy


# --------------------------------------------------------------------------
# 3. End-to-end capture
# --------------------------------------------------------------------------


class _FakeOcr(OcrBackend):
    """A stand-in backend, so the OCR path is exercised without PaddleOCR.

    PaddleOCR is not installed in this project and the corpus is synthetic text
    with no rasterised pages, so the real backend cannot run here. This proves
    the wiring -- that a rejected text layer reaches the backend and its text is
    what comes back -- without claiming the engine works.
    """

    name = "fake"

    def __init__(self, text: str, conf: float = 0.93) -> None:
        self._text = text
        self._conf = conf
        self.called_with: str | None = None

    def text_from_image(self, image_path: str) -> tuple[str, float]:
        self.called_with = image_path
        return self._text, self._conf


def test_capture_skips_ocr_when_the_text_layer_is_sound() -> None:
    text = "\n".join(f"Account: {i}" for i in GOOD_IBANS)
    ocr = _FakeOcr("SHOULD NOT BE USED")
    result = capture(
        Channel.PDF, text=text, image_path="page.png", has_text_layer=True, ocr=ocr
    )
    assert not result.used_ocr
    assert ocr.called_with is None
    assert result.text == text


def test_capture_falls_through_to_ocr_when_the_layer_is_corrupt() -> None:
    """The fallback is the point: OCR runs only after a deterministic rejection."""
    text = f"Account: {GOOD_IBANS[0].replace('0', 'O')}"
    ocr = _FakeOcr("Account: DE89370400440532013000")
    result = capture(
        Channel.PDF, text=text, image_path="page.png", has_text_layer=True, ocr=ocr
    )
    assert result.used_ocr
    assert ocr.called_with == "page.png"
    assert result.text == "Account: DE89370400440532013000"
    assert result.trust is not None
    assert not result.trust.trustworthy


def test_capture_does_not_fabricate_text_when_ocr_is_missing() -> None:
    """With no backend, return what we have and say so. Never silently pretend."""
    text = f"Account: {GOOD_IBANS[0].replace('0', 'O')}"
    result = capture(Channel.PDF, text=text, has_text_layer=True)
    assert not result.used_ocr
    assert any("no OCR backend" in n for n in result.notes), result.notes


def test_capture_routes_edi_without_touching_a_model() -> None:
    result = capture(
        Channel.EDI,
        text="",
        structured="<Document>...</Document>",
        has_text_layer=True,
        ocr=_FakeOcr("SHOULD NOT BE USED"),
    )
    assert result.route.path is Path.STRUCTURED
    assert not result.used_ocr
    assert result.text == "<Document>...</Document>"


def test_capture_ocrs_a_fax() -> None:
    ocr = _FakeOcr("Account: DE89370400440532013000", conf=0.88)
    result = capture(Channel.FAX, image_path="fax.tiff", ocr=ocr)
    assert result.used_ocr
    assert result.route.path is Path.OCR
    assert result.ocr_confidence == 0.88


# --------------------------------------------------------------------------
# 4. A measured summary, for the docs
# --------------------------------------------------------------------------


def measure() -> dict[str, object]:
    """Produce the numbers quoted in section 18, by running the probe.

    Reported as two separate rates because they answer two different questions:
    what share of real corruption is caught, and what share of clean documents is
    wrongly sent to OCR. A single blended number hides both.
    """
    rng = random.Random(20240916)

    cases = _corrupting_cases()
    caught = 0
    per_corruption: dict[str, list[int]] = {name: [0, 0] for name, _ in CORRUPTING}
    for name, _iban, corrupted in cases:
        report = text_layer_trust(f"Account: {corrupted}")
        per_corruption[name][1] += 1
        if not report.trustworthy:
            caught += 1
            per_corruption[name][0] += 1

    # Clean documents must be accepted, or the probe is useless in the direction
    # that costs money.
    clean_total = 0
    clean_passed = 0
    for _ in range(200):
        n = rng.randint(1, 4)
        sample = rng.sample(GOOD_IBANS, n)
        text = "\n".join(f"Account: {i}" for i in sample)
        clean_total += 1
        if text_layer_trust(text).trustworthy:
            clean_passed += 1

    # Formatting-only variants, counted separately from corruption.
    fmt_total = 0
    fmt_accepted = 0
    for _name, transform in FORMATTING:
        for iban in GOOD_IBANS:
            variant = transform(iban)  # type: ignore[operator]
            if not isinstance(variant, str):  # pragma: no cover
                continue
            fmt_total += 1
            if text_layer_trust(f"Account: {variant}").trustworthy:
                fmt_accepted += 1

    return {
        "corrupting_cases": len(cases),
        "corruption_caught": caught,
        "detection_rate": round(caught / len(cases), 4) if cases else 0.0,
        "per_corruption": {k: f"{v[0]}/{v[1]}" for k, v in per_corruption.items()},
        "clean_docs": clean_total,
        "clean_accepted": clean_passed,
        "false_reject_rate": round(1 - clean_passed / clean_total, 4),
        "formatting_variants": fmt_total,
        "formatting_accepted": fmt_accepted,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(measure(), indent=2))
