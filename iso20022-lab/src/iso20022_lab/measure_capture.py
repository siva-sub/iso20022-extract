"""Measured comparison: capture routes versus regex-on-exact-text.

The question this answers is the one that decides the whole architecture:

    Does adding OCR make field extraction better or worse?

The intuitive answer -- "OCR reads the document, so it can only help" -- is
wrong, and wrong in the specific way that costs money. Recognition is lossy. It
runs a probabilistic model over pixels and emits characters, and the characters
it emits are sometimes not the ones that were printed. Extraction then operates
on those characters, and a rules engine that is 100% correct on exact text will
faithfully extract a corrupted value.

So the correct baseline is not "regex versus OCR". It is:

    A) rules on the EXACT text          -- what a text layer or a typed body gives
    B) rules on the RECOGNISED text     -- what a fax or a scan gives

A is the ceiling. B is what you get when you point a camera at the document. The
gap between them is the **recognition tax**, and this module measures it instead
of asserting it.

WHAT THE MEASUREMENT IS FOR

It is not a benchmark of PP-OCRv6. It is a decision procedure for the routing
table in `ingest.py`: it produces the evidence for when OCR is worth paying for.
Two things follow from it and both are in PROCESS.md 18.11:

1. OCR is only justified when there is no text layer. For a document that already
   carries exact characters, recognition can only subtract information.
2. A checksummed field is not a verified field. Some recognition damage is
   caught by the validators and some is invisible to them, and the difference
   between those two cases is not visible from the accuracy number alone -- it
   is visible in *which fields* break.

Both are why this reports per-field damage rather than one aggregate score. An
aggregate hides the case that matters, because the field that breaks silently is
outnumbered by the fields that break loudly.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from iso20022_lab.baseline import RuleExtractor
from iso20022_lab.ingest import OcrBackend

if TYPE_CHECKING:
    from iso20022_lab.corpus import Document


@dataclass
class FieldOutcome:
    """One field, on one document, under one capture route."""

    key: str
    truth: str
    got: str

    @property
    def correct(self) -> bool:
        return self.truth == self.got


@dataclass
class Damage:
    """One wrong field, attributed to a cause.

    Three causes, and they need different fixes, so collapsing them into one
    "wrong" count makes the result unactionable:

    - `role`     the value is a real value from *this document*, in the wrong
                 field. Recognition usually causes it indirectly: it damages a
                 label, the extractor loses its anchor, and the positional
                 fallback picks the other party's value. The value itself is
                 valid, current, and belongs to the same payment.
    - `missing`  nothing extracted. A *visible* failure: the document is
                 incomplete, fails the completeness check, and goes to review.
                 Not silent, and must not be counted as silent.
    - `caught`   the characters changed and a validator rejects the result, so
                 the pipeline flags it for review.
    - `silent`   the characters changed and nothing rejects it: no checksum
                 applies to this field, or the corrupted form still validates.
                 This is the only unstoppable class, and the one worth
                 engineering against.
    """

    key: str
    truth: str
    got: str
    cause: str


@dataclass
class RouteScore:
    """Aggregate for one route across all documents."""

    name: str
    docs: int = 0
    fields: int = 0
    correct: int = 0
    documents_fully_correct: int = 0
    seconds: float = 0.0
    # Damage attributed per canonical key, so a single quietly-wrong field is
    # visible instead of averaged away.
    missed_by_field: Counter[str] = field(default_factory=Counter)
    damage: list[Damage] = field(default_factory=list)

    def caused(self, cause: str) -> list[Damage]:
        return [d for d in self.damage if d.cause == cause]

    @property
    def role_errors(self) -> int:
        return len(self.caused("role"))

    @property
    def missing_errors(self) -> int:
        return len(self.caused("missing"))

    @property
    def silent_errors(self) -> int:
        return len(self.caused("silent"))

    @property
    def caught_errors(self) -> int:
        return len(self.caused("caught"))

    @property
    def visible_errors(self) -> int:
        """Errors some existing check already catches."""
        return self.missing_errors + self.caught_errors

    @property
    def invisible_errors(self) -> int:
        """Errors nothing catches: wrong role, or wrong characters that validate."""
        return self.role_errors + self.silent_errors

    @property
    def field_accuracy(self) -> float:
        return self.correct / self.fields if self.fields else 0.0

    @property
    def document_stp(self) -> float:
        return self.documents_fully_correct / self.docs if self.docs else 0.0

    def render(self) -> str:
        return (
            f"  {self.name:<26} fields {self.field_accuracy:7.2%}  "
            f"STP {self.document_stp:7.2%}  "
            f"{self.correct}/{self.fields}  {self.seconds:.1f}s"
        )


@dataclass
class CaptureReport:
    routes: list[RouteScore] = field(default_factory=list)

    def by_name(self, name: str) -> RouteScore | None:
        return next((r for r in self.routes if r.name == name), None)

    def render(self) -> str:
        lines = ["", "=" * 78, "CAPTURE ROUTE COMPARISON", "=" * 78]
        lines.extend(r.render() for r in self.routes)

        exact = self.by_name("rules on exact text")
        ocr = self.by_name("rules on recognised text")
        if exact and ocr:
            lines.append("")
            lines.append("RECOGNITION TAX (exact minus recognised)")
            lines.append(
                f"  field accuracy  {exact.field_accuracy - ocr.field_accuracy:+.2%}"
            )
            lines.append(f"  document STP    {exact.document_stp - ocr.document_stp:+.2%}")
            slowdown = ocr.seconds / exact.seconds if exact.seconds else 0.0
            lines.append(f"  wall clock      {slowdown:.0f}x slower")

        if ocr and ocr.missed_by_field:
            lines.append("")
            lines.append("WHERE RECOGNITION BROKE FIELDS (recognised route)")
            for key, n in ocr.missed_by_field.most_common():
                lines.append(f"  {key:<24} {n} miss(es)")

        if ocr and ocr.damage:
            lines.append("")
            lines.append("DAMAGE BY CAUSE (recognised route)")
            lines.append(
                f"  role    {ocr.role_errors:>4}  a real value from this document put in\n"
                "                the wrong field"
            )
            lines.append(
                f"  missing {ocr.missing_errors:>4}  nothing extracted; incomplete, so review sees it"
            )
            lines.append(
                f"  caught  {ocr.caught_errors:>4}  characters changed and a validator rejects it"
            )
            lines.append(
                f"  silent  {ocr.silent_errors:>4}  characters changed and nothing rejects it"
            )
            lines.append("")
            lines.append(f"  visible to existing checks : {ocr.visible_errors:>4}")
            lines.append(
                f"  INVISIBLE to all of them   : {ocr.invisible_errors:>4}"
                "   <- these reach a payment"
            )

            silent = ocr.caused("silent")
            if silent:
                lines.append("")
                lines.append("THE SILENT ONES (nothing in the pipeline flags these)")
                seen: set[tuple[str, str, str]] = set()
                for d in silent:
                    if (d.key, d.truth, d.got) in seen:
                        continue
                    seen.add((d.key, d.truth, d.got))
                    lines.append(f"  {d.key:<22} {d.truth}  ->  {d.got}")
                    if len(seen) >= 6:
                        break

        lines.append("")
        return "\n".join(lines)


def _canonical_values(doc: Document) -> dict[str, str]:
    """Truth values under canonical (ISO 20022) key names."""
    truth = getattr(doc, "truth", None)
    if truth:
        return {str(k): str(v) for k, v in truth.items()}
    values = getattr(doc, "values", None)
    if values:
        return {str(k): str(v) for k, v in values.items()}
    return {}


def _is_plausible(key: str, value: str) -> bool:
    """Would a deterministic validator accept this wrong value?

    This is the whole point of the report. A value that fails its checksum is a
    caught error and the pipeline routes it to review. A value that passes is
    routed to the payment, so `plausible_wrong` counts the second kind only.
    """
    try:
        from iso20022_lab.validators import bic_valid, iban_valid
    except ImportError:
        return False
    if "IBAN" in key:
        return iban_valid(value)
    if "BIC" in key:
        return bic_valid(value)
    return False


def measure(
    documents: Sequence[Document],
    *,
    ocr: OcrBackend | None,
    renderer: Callable[[str], str | Path] | None = None,
) -> CaptureReport:
    """Run both routes over the same documents and compare.

    `Sequence` rather than `list`: this only reads the collection, and a list of
    a concrete type is not a list of `object` (lists are invariant), so the wider
    annotation would force every caller to widen its own data for no benefit.

    The text route is genuinely free -- no model, no pixels -- so its timing is
    the cost of the rules engine alone and the ratio between the routes is the
    cost of recognition.
    """
    report = CaptureReport()
    exact = RouteScore(name="rules on exact text")
    recognised = RouteScore(name="rules on recognised text")
    extractor = RuleExtractor()

    for doc in documents:
        text = getattr(doc, "text", "")
        truth = _canonical_values(doc)
        if not text or not truth:
            continue

        # --- A) rules on the exact text ---
        t0 = time.monotonic()
        got = extractor.extract(text)
        exact.seconds += time.monotonic() - t0
        _score(exact, truth, _as_mapping(got))

        if ocr is None:
            continue

        # --- B) rules on the recognised text ---
        image_path = _render(text, renderer)
        if image_path is None:
            continue
        t0 = time.monotonic()
        ocr_text, _conf = ocr.text_from_image(str(image_path))
        ocr_got = extractor.extract(ocr_text)
        recognised.seconds += time.monotonic() - t0
        _score(recognised, truth, _as_mapping(ocr_got))

    report.routes = [exact] + ([recognised] if ocr is not None else [])
    return report


def _as_mapping(extraction: object) -> dict[str, str]:
    """Rule extractor output as a plain mapping."""
    values = getattr(extraction, "values", None)
    if isinstance(values, dict):
        return {str(k): str(v) for k, v in values.items()}
    fields_attr = getattr(extraction, "fields", None)
    if isinstance(fields_attr, dict):
        return {str(k): str(v) for k, v in fields_attr.items()}
    if isinstance(extraction, dict):
        return {str(k): str(v) for k, v in extraction.items()}
    return {}


def _score(route: RouteScore, truth: dict[str, str], got: dict[str, str]) -> None:
    """Attribute every wrong field to a cause.

    The role check comes first and is what makes this measurement trustworthy. An
    earlier version classified any wrong-but-valid value as recognition damage,
    and the output showed `ES684...` turning into `ES091...` and `ES091...`
    turning into `ES684...` -- the same two values, swapped. Recognition cannot
    swap two valid account numbers out of one document; only a role collapse can.
    Counting that as OCR damage inflated the recognition tax and would have sent
    the work to the wrong component.

    No `doc` parameter. An earlier signature took one and never read it, which
    made every caller invent an argument -- and the tests passed `None`, which is
    exactly the kind of unused slot that invites a caller to pass something wrong
    later because nothing would notice.
    """
    route.docs += 1
    hits = 0
    truth_values = set(truth.values())
    for key, want in truth.items():
        route.fields += 1
        mine = got.get(key, "")
        if mine == want:
            route.correct += 1
            hits += 1
            continue
        route.missed_by_field[key] += 1
        if not mine:
            cause = "missing"
        elif mine in truth_values:
            cause = "role"
        elif _is_plausible(key, mine):
            cause = "silent"
        else:
            cause = "caught"
        route.damage.append(Damage(key=key, truth=want, got=mine, cause=cause))
    if hits == len(truth):
        route.documents_fully_correct += 1


def _render(
    text: str, renderer: Callable[[str], str | Path] | None, *, width: int = 1240
) -> Path | None:
    """Rasterise document text the way a fax would arrive, for route B.

    Renders *every* line, sizing the image to the text. An earlier version drew a
    fixed 14 lines into a fixed 420-pixel box, which silently truncated longer
    documents -- and a truncated document fails route B for a reason that has
    nothing to do with recognition. Measuring a crop and calling it a recognition
    error would have made the recognition tax look larger than it is.
    """
    if renderer is not None:
        return Path(renderer(text))
    import tempfile

    from PIL import Image, ImageDraw

    lines = text.split("\n")
    line_height = 26
    padding = 24
    height = max(120, padding * 2 + line_height * len(lines))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill="black")
        y += line_height
    out = Path(tempfile.mkdtemp(prefix="capture-measure-")) / "page.png"
    image.save(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ocr", choices=["none", "tiny", "medium"], default="tiny")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    from iso20022_lab.corpus import build_corpus
    from iso20022_lab.synth import generate_many

    # Same construction the training set uses, so route A and route B are measured
    # on documents of the kind the pipeline actually sees.
    values = generate_many(args.docs, seed=args.seed)
    corpus = build_corpus(values, per_seed=1, seed=args.seed)
    docs = corpus.documents
    print(f"documents: {len(docs)} (seed {args.seed})")
    print(corpus.summary())

    backend: OcrBackend | None = None
    if args.ocr != "none":
        from iso20022_lab.ocr import build_medium, build_tiny

        backend = build_tiny() if args.ocr == "tiny" else build_medium()
        print(f"OCR backend: {type(backend).__name__}")

    report = measure(docs, ocr=backend)
    print(report.render())

    if args.json_out:
        payload = {
            r.name: {
                "docs": r.docs,
                "fields": r.fields,
                "correct": r.correct,
                "field_accuracy": r.field_accuracy,
                "document_stp": r.document_stp,
                "seconds": r.seconds,
                "missed_by_field": dict(r.missed_by_field),
                "role_errors": r.role_errors,
                "missing_errors": r.missing_errors,
                "caught_errors": r.caught_errors,
                "silent_errors": r.silent_errors,
                "visible_errors": r.visible_errors,
                "invisible_errors": r.invisible_errors,
                "damage": [
                    {"key": d.key, "truth": d.truth, "got": d.got, "cause": d.cause}
                    for d in r.damage
                ],
            }
            for r in report.routes
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
