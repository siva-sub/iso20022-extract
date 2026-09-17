"""Inbound capture: routing email, PDF, portal, EDI and fax into one pipeline.

PROCESS.md section 18 documents this. The short version: there are five inbound
channels, and **OCR is needed by only two of them**. Treating OCR as the default
front door is the most common way to make this system slow, lossy and expensive
for no gain.

    channel            what arrives              needs OCR?
    ------------------------------------------------------------------
    email body         plain/HTML text           no
    email attachment   route by MIME type        depends
    PDF, text layer    extractable characters    no, if the layer is sound
    PDF, scanned       pixels only               yes
    portal upload      usually HTML->PDF         no, if the layer is sound
    EDI (MT101/pain)   already structured        no -- parse it
    fax                pixels only               yes

The design decision that makes this cheap is in `text_layer_trust`. A PDF's
embedded text layer is frequently *present but wrong*: ligature mapping
failures, CID font substitutions, wrong reading order in multi-column layouts.
Detecting that normally means running OCR and diffing, which defeats the point.

Instead this module reuses the **deterministic validators already used to gate
the output** -- IBAN mod-97, BIC shape, minor units -- as an input-quality
probe. If the text layer yields account numbers that fail their checksum, the
layer is corrupt, and the document is routed to OCR. If they pass, OCR is
skipped. The validators earn their keep twice, and the probe is deterministic,
auditable and free.

This is not a heuristic guess. ISO 13616 IBAN check digits catch essentially
every single-character corruption, and a BIC's fixed shape catches most of the
rest. Section 18 carries the measured detection rates.

ON THE PADDLEOCR INTEGRATION, HONESTLY:

`PaddleOcr` below is written against the PaddleOCR 3.x API (PP-OCRv6 detection
and recognition, PP-StructureV3 for layout). It is **not exercised by this
project's tests** -- PaddleOCR is not installed here and the corpus is synthetic
text with no rasterised pages, so there is nothing to recognise. It is the
integration point, shaped to the documented API, and it must be tested against
real scanned documents before anyone relies on it. Section 18 says which claims
are measured and which are not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from iso20022_lab.validators import bic_country_valid, iban_valid

# --------------------------------------------------------------------------
# Channels
# --------------------------------------------------------------------------


class Channel(str, Enum):
    """Where a payment instruction entered the building."""

    EMAIL_BODY = "email-body"
    EMAIL_ATTACHMENT = "email-attachment"
    PDF = "pdf"
    PORTAL = "portal"
    EDI = "edi"
    FAX = "fax"


class Path(str, Enum):
    """What the pipeline does with it."""

    TEXT = "text"  # usable characters already; no OCR
    STRUCTURED = "structured"  # already a message; parse, never OCR
    OCR = "ocr"  # pixels only; recognition required
    PROBE = "probe"  # text layer exists; validate before trusting it


# --------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Route:
    """The routing decision, with the reason attached.

    The reason is not decoration. When a document is misrouted, the reason is
    what tells you which rule to change.
    """

    channel: Channel
    path: Path
    reason: str


def route(
    channel: Channel,
    *,
    has_text_layer: bool = False,
    is_structured: bool = False,
) -> Route:
    """Decide the capture path for one inbound document.

    Deliberately takes only facts that can be determined cheaply -- MIME type,
    whether a text layer exists, whether the body is already a message. Anything
    expensive (OCR, layout analysis) happens after this, and only if selected.
    """
    if is_structured or channel is Channel.EDI:
        # MT101 and pain.001 are already messages. Running OCR over a structured
        # payment is pure loss: it converts correct data into a guess.
        return Route(channel, Path.STRUCTURED, "already a structured message")

    if channel is Channel.EMAIL_BODY:
        # The instruction is in the mail body. There is no image and no layout,
        # so there is nothing for an OCR or a layout model to do.
        return Route(channel, Path.TEXT, "body text, no raster content")

    if channel in (Channel.FAX, Channel.EMAIL_ATTACHMENT) and not has_text_layer:
        return Route(channel, Path.OCR, "pixels only")

    if has_text_layer:
        # The interesting case. A text layer exists, but existing is not the
        # same as being right. Probe it before trusting it.
        return Route(channel, Path.PROBE, "text layer present, trust unverified")

    return Route(channel, Path.OCR, "no text layer")


# --------------------------------------------------------------------------
# The trust probe
# --------------------------------------------------------------------------

# IBANs and BICs as they appear in free text. Deliberately loose about spacing:
# a text layer that has dropped or added spaces is one of the failures this is
# supposed to catch, so the extractor must not depend on clean spacing.
#
# The separators are literal spaces, NOT \s. This was a real bug: `\s?` matches
# a newline, so on text laid out as one field per line the IBAN pattern ran
# across the line break and swallowed the next label -- `DE89...3000` followed
# by `\nAccount` matched as one 22-character token, failed its checksum, and the
# whole document was declared corrupt. Measured effect: **77% of clean documents
# were rejected** and sent to OCR for nothing. Spacing inside an IBAN is a
# literal space; a newline means the field ended.
#
# IGNORECASE is load-bearing, not cosmetic. A text layer that has lowercased the
# document is a real failure mode, and a case-sensitive pattern simply finds
# nothing, which the probe would then report as "no checksummed fields" rather
# than as a verdict. A lowercased document would be routed to OCR on the wrong
# grounds and the report would not say why.
_IBAN_RE = re.compile(r"\b([A-Z]{2} ?\d{2}(?: ?[A-Z0-9]){11,30})\b", re.IGNORECASE)
_BIC_RE = re.compile(
    r"\b([A-Z]{4} ?[A-Z]{2} ?[A-Z0-9]{2}(?: ?[A-Z0-9]{3})?)\b", re.IGNORECASE
)

# A BIC-shaped token only counts as evidence when a BIC label precedes it.
#
# Shape alone is not enough, and neither is shape plus a real country code. This
# was measured, not theorised: `attached` matches the BIC shape AND its positions
# 5-6 are `CH`, which is genuinely Switzerland. Adding country validation
# therefore did not reject it. It is structurally indistinguishable from a BIC by
# pattern matching, because it IS one by pattern matching.
#
# The consequences of counting it are asymmetric and bad. A false valid raises the
# trust score, so prose scores as a verified text layer -- confidence manufactured
# in a document where nothing was checked. Requiring context makes the signal
# mean what it claims: a token behind a BIC label is strong evidence, a token in
# the middle of a sentence is a coincidence.
_BIC_CONTEXT = re.compile(
    r"(?:\bBIC\b|\bBICFI\b|\bSWIFT\b|\bBLZ\b|BANK\s+IDENTIFIER|BANK\s+CODE"
    r"|IDENTIFIER\s+CODE)\s*[:#]?\s*$",
    re.IGNORECASE,
)


@dataclass
class TrustReport:
    """What the probe found in a text layer."""

    ibans_found: int = 0
    ibans_valid: int = 0
    bics_found: int = 0
    bics_valid: int = 0
    score: float = 0.0
    trustworthy: bool = False
    reason: str = ""
    failed_samples: list[str] = field(default_factory=list)

    def as_line(self) -> str:
        return (
            f"score {self.score:.2f}  iban {self.ibans_valid}/{self.ibans_found}  "
            f"bic {self.bics_valid}/{self.bics_found}  "
            f"{'TRUST' if self.trustworthy else 'REJECT'}  ({self.reason})"
        )


def _strip(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()


def _has_bic_context(text: str, start: int) -> bool:
    """Whether a BIC-shaped match at `start` is preceded by a BIC label.

    Only the current line is examined. A BIC label on a different field is not
    evidence about this token, and allowing the search to run across lines would
    reopen the same accidental-match hole from the other side.
    """
    line_start = text.rfind("\n", 0, start) + 1
    return bool(_BIC_CONTEXT.search(text[line_start:start]))


def text_layer_trust(text: str, threshold: float = 0.99) -> TrustReport:
    """Decide whether an extracted PDF text layer is fit to use.

    The probe is checksum-based. An IBAN's two check digits (ISO 13616, mod-97)
    are correct in the source document by definition, so any text layer that
    produces IBANs failing mod-97 has corrupted them. There is no need to know
    what the right value was -- the checksum says it is wrong.

    A BIC is 8 or 11 characters in a fixed shape, so a corrupt read usually
    breaks the shape.

    Default threshold is 0.99, not 1.0. A single bad IBAN in a long document is
    not by itself proof of a bad text layer, and rejecting the whole layer on one
    outlier would send clean documents to OCR. The threshold is a parameter
    because it is a business decision about which error is worse: a missed
    corruption sends a wrong value downstream, an over-eager rejection costs
    OCR time on a document that did not need it.

    Returns a report; `trustworthy=False` means route to OCR.
    """
    report = TrustReport()

    for raw in _IBAN_RE.findall(text):
        candidate = _strip(raw)
        # A bare 2-letter + 2-digit + alnum run also matches BIC-shaped and
        # reference-shaped tokens. Only count things that are IBAN-length.
        if not (15 <= len(candidate) <= 34):
            continue
        report.ibans_found += 1
        if iban_valid(candidate):
            report.ibans_valid += 1
        elif len(report.failed_samples) < 5:
            report.failed_samples.append(candidate)

    for match in _BIC_RE.finditer(text):
        raw = match.group(1)
        candidate = _strip(raw)
        if len(candidate) not in (8, 11):
            continue
        # Skip anything already counted as an IBAN; the patterns overlap.
        if candidate in report.failed_samples:
            continue
        # An unlabelled match is a coincidence, not evidence. See _BIC_CONTEXT.
        if not _has_bic_context(text, match.start()):
            continue
        report.bics_found += 1
        # bic_country_valid, not bic_valid. Shape alone accepts English words:
        # "attached" is CH, "beneficiary" is FI, "instruction" is RU -- all real
        # country codes, so the country check is necessary. It is not sufficient
        # (CH really is a country), which is why the context requirement above
        # carries the weight and this one filters what gets through.
        if bic_country_valid(candidate):
            report.bics_valid += 1
        elif len(report.failed_samples) < 5:
            report.failed_samples.append(candidate)

    checked = report.ibans_found + report.bics_found
    if checked == 0:
        # Nothing checkable to go on. This is the honest gap: the probe can only
        # speak about documents that carry checksummed fields. A document with no
        # IBAN and no BIC gets no verdict from this function, and the caller has
        # to decide on other grounds.
        report.score = 0.0
        report.trustworthy = False
        report.reason = "no checksummed fields to probe"
        return report

    report.score = (report.ibans_valid + report.bics_valid) / checked
    report.trustworthy = report.score >= threshold
    if report.trustworthy:
        report.reason = f"all {checked} checksummed fields valid"
    else:
        report.reason = (
            f"{checked - report.ibans_valid - report.bics_valid} of {checked} "
            f"checksummed fields invalid"
        )
    return report


# --------------------------------------------------------------------------
# OCR backend
# --------------------------------------------------------------------------


class OcrBackend:
    """Interface an OCR implementation must satisfy.

    Kept to one method on purpose. Everything the rest of the pipeline needs to
    know about an OCR engine is "give me text for these pixels".
    """

    name: str = "base"

    def text_from_image(self, image_path: str) -> tuple[str, float]:
        """Return (text, mean recognition confidence in 0..1)."""
        raise NotImplementedError


class PaddleOcr(OcrBackend):
    """PaddleOCR 3.x behind `OcrBackend`.

    Written against the documented 3.x API. NOT exercised by this project's
    tests -- see the module docstring and PROCESS.md section 18. Treat it as the
    integration point to verify on real scans, not as tested code.

    Why this engine and not a document VLM: PP-OCRv6 is a sub-100M-parameter
    specialist, and PaddleOCR's own technical report argues that specialists of
    this size rival billion-parameter VLMs at OCR. At 100k documents/month, a
    local sub-100MB model on CPU is a vastly better operating point than a
    hosted multimodal call, and it keeps payment data on-premises.

    `use_doc_orientation_classify` and `use_doc_unwarping` are left on
    deliberately: faxed and photographed instructions are routinely skewed, and
    un-warping before recognition is cheaper than a recognition model that can
    tolerate rotation.

    PP-StructureV3 is the variant to reach for when layout matters -- multi-column
    remittances, tables of invoices -- because it returns reading order and
    structured blocks rather than a flat string. It is slower, so it is selected
    per-document rather than used as the default.
    """

    name = "paddleocr"

    def __init__(self, structured: bool = False, lang: str = "en") -> None:
        self.structured = structured
        self.lang = lang
        self._pipeline = None

    def _load(self) -> Any:
        """Import and construct lazily.

        PaddleOCR is a heavy optional dependency. Importing it at module scope
        would make `ingest` unimportable in any environment that only needs the
        router -- which is most of them, including this project's test run.
        """
        if self._pipeline is not None:
            return self._pipeline

        if self.structured:
            # pyright: ignore[reportMissingImports] -- paddleocr is an optional
            # dependency, deliberately not installed in this project. See the
            # module docstring: this path is unexercised here.
            from paddleocr import PPStructureV3  # pyright: ignore[reportMissingImports]

            self._pipeline: Any = PPStructureV3()
        else:
            from paddleocr import PaddleOCR  # pyright: ignore[reportMissingImports]

            self._pipeline = PaddleOCR(
                lang=self.lang,
                use_doc_orientation_classify=True,
                use_doc_unwarping=True,
            )
        return self._pipeline

    def text_from_image(self, image_path: str) -> tuple[str, float]:
        """Return (text, mean recognition confidence).

        PaddleOCR 3.x returns a list of result objects, each carrying parallel
        `rec_texts` and `rec_scores` lists. Confidence is averaged and returned
        rather than discarded: a low mean score is a signal to route the document
        to a human instead of quietly forwarding a shaky read into extraction.
        """
        pipeline = self._load()
        results = pipeline.predict(image_path)

        lines: list[str] = []
        scores: list[float] = []
        for res in results:
            data = getattr(res, "json", None)
            payload = data.get("res", data) if isinstance(data, dict) else {}
            rec_texts = payload.get("rec_texts", [])
            rec_scores = payload.get("rec_scores", [])
            lines.extend(rec_texts)
            scores.extend(float(s) for s in rec_scores)

        text = "\n".join(lines)
        mean_conf = sum(scores) / len(scores) if scores else 0.0
        return text, mean_conf


# --------------------------------------------------------------------------
# The full path
# --------------------------------------------------------------------------


@dataclass
class CaptureResult:
    """What came back from the capture stage."""

    route: Route
    text: str
    used_ocr: bool = False
    ocr_confidence: float = 1.0
    trust: TrustReport | None = None
    notes: list[str] = field(default_factory=list)


def capture(
    channel: Channel,
    *,
    text: str = "",
    image_path: str | None = None,
    structured: str | None = None,
    has_text_layer: bool = False,
    ocr: OcrBackend | None = None,
) -> CaptureResult:
    """Run one document through capture, returning text ready for extraction.

    Order matters and is the whole point: cheap deterministic checks first, OCR
    last and only when selected. A document that arrives as text or as a
    structured message never touches a model.
    """
    decision = route(channel, has_text_layer=has_text_layer, is_structured=bool(structured))

    if decision.path is Path.STRUCTURED:
        return CaptureResult(
            route=decision,
            text=structured or text,
            notes=["parsed as a message; no recognition performed"],
        )

    if decision.path is Path.TEXT:
        return CaptureResult(route=decision, text=text)

    if decision.path is Path.PROBE:
        report = text_layer_trust(text)
        if report.trustworthy:
            return CaptureResult(
                route=decision,
                text=text,
                trust=report,
                notes=["text layer accepted without OCR"],
            )
        # The layer failed its own checksum. Fall through to OCR, but keep the
        # report -- it explains why this document cost OCR time.
        if ocr is None or image_path is None:
            return CaptureResult(
                route=Route(channel, Path.OCR, f"text layer failed: {report.reason}"),
                text=text,
                trust=report,
                notes=[
                    "text layer rejected but no OCR backend supplied; "
                    "returning unverified text"
                ],
            )
        ocr_text, conf = ocr.text_from_image(image_path)
        return CaptureResult(
            route=Route(channel, Path.OCR, f"text layer failed: {report.reason}"),
            text=ocr_text,
            used_ocr=True,
            ocr_confidence=conf,
            trust=report,
            notes=[f"rejected text layer: {report.reason}"],
        )

    # Path.OCR
    if ocr is None or image_path is None:
        return CaptureResult(
            route=decision,
            text=text,
            notes=["OCR required but no backend supplied"],
        )
    ocr_text, conf = ocr.text_from_image(image_path)
    return CaptureResult(route=decision, text=ocr_text, used_ocr=True, ocr_confidence=conf)
