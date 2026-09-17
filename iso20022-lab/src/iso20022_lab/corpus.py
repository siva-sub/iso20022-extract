"""Document corpus with exact ground truth, for the Phase 3 measurement.

PROCESS.md 3.3 says the STP claim needs "exceptions per day, minutes per
exception, loaded cost per hour" and that until those exist the claim is a
hypothesis. This module supplies the first of the three: a corpus on which an
exception rate can actually be measured.

There is no large public corpus of payment documents with paired ground truth,
for the same reason there is no large public MT corpus: both carry PII. So the
corpus is built by rendering REAL field values -- harvested from the real MT/MX
fixtures under data/ -- into documents, recording exactly what was rendered.

Three honesty constraints, all load-bearing:

1. The values are real; the prose is synthetic. Documents rendered from variant
   pools have less linguistic variety than real correspondence, so accuracy
   measured here is an UPPER BOUND for both extractors.
2. The renderer and the baseline must not share conventions. If the baseline
   were tuned to the exact surface forms this renderer emits, the measurement
   would be circular. The renderer draws formats from independent pools, and
   the baseline gets a competent normaliser for each, so the failures that
   remain are semantic rather than cosmetic.
3. `truth` holds CANONICAL values, not surface forms. The extractor's job is to
   recover "1250.00" from "1.250,00", because that is what the message needs. A
   test that accepted the surface form would not measure normalisation at all.

Document keys are the pipeline's own key space (`distill.field_key` output), so
one evaluator scores the rule baseline, the teacher, and the student without a
translation layer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from iso20022_lab.mt import mt103_to_dict, mx_to_dict, parse_mt103

# --------------------------------------------------------------------------
# Field vocabulary, in the pipeline key space.
# --------------------------------------------------------------------------

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "Cdtr_Nm": ("Beneficiary", "Beneficiary Name", "Payee", "Creditor"),
    "CdtrAcct_IBAN": ("Beneficiary IBAN", "IBAN", "Account", "A/C"),
    "CdtrAgt_BICFI": ("Beneficiary BIC", "BIC", "SWIFT", "SWIFT/BIC"),
    "Dbtr_Nm": ("Payer", "Debtor", "Account Holder", "From"),
    "DbtrAcct_IBAN": ("Payer IBAN", "Debit Account", "From Account", "A/C"),
    "Amt_InstdAmt": ("Amount", "Value", "Amount Due", "Total"),
    "PmtId_EndToEndId": ("Payment Reference", "Reference", "Ref", "End-to-End ID"),
    "ReqdExctnDt_Dt": ("Execution Date", "Value Date", "Settlement Date", "Due"),
    "RmtInf_Ustrd": ("Remittance Information", "Details", "Narrative"),
}

# The currency is rendered inline with the amount, so it is never its own line.
INLINE_KEYS: frozenset[str] = frozenset({"Amt_InstdAmt_Ccy"})

# Distractors are what make the task hard. A document containing only the target
# fields is a parsing exercise; a real one contains numbers that look exactly
# like money and references that look exactly like the reference.
DISTRACTOR_LABELS: tuple[str, ...] = (
    "Invoice No",
    "Customer No",
    "VAT Reg",
    "Phone",
    "Order Ref",
    "PO Number",
    "Sort Code",
)


@dataclass(frozen=True)
class Document:
    """A rendered document plus the canonical values it was built from."""

    doc_id: str
    text: str
    truth: dict[str, str]
    surface: dict[str, str]
    difficulty: str
    notes: tuple[str, ...] = ()


@dataclass
class RenderConfig:
    """Knobs that control how hostile a document is."""

    use_label_synonyms: bool = False
    european_numbers: bool = False
    varied_dates: bool = False
    distractors: int = 0
    omit_label: tuple[str, ...] = ()
    bury_in_prose: tuple[str, ...] = ()


DIFFICULTIES: dict[str, RenderConfig] = {
    # Labels canonical, one format throughout, nothing else numeric.
    "clean": RenderConfig(),
    # Labels vary, formats vary, some noise. What a real inbox looks like.
    "messy": RenderConfig(
        use_label_synonyms=True,
        european_numbers=True,
        varied_dates=True,
        distractors=2,
    ),
    # The fields that matter most are stated in running prose with no label at
    # all, alongside heavy noise. Label-anchored rules have no anchor here, and
    # that is the point: it is where the two approaches must separate.
    "hostile": RenderConfig(
        use_label_synonyms=True,
        european_numbers=True,
        varied_dates=True,
        distractors=4,
        bury_in_prose=("CdtrAcct_IBAN", "Amt_InstdAmt"),
    ),
}


def _fmt_amount(value: str, cfg: RenderConfig) -> str:
    """Render an amount in US or European convention.

    The baseline is given a competent normaliser for both, so this is not where
    it is meant to fail.
    """
    try:
        num = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return value
    if cfg.european_numbers:
        whole, _, frac = f"{num:,.2f}".partition(".")
        return f"{whole.replace(',', '.')},{frac}"
    return f"{num:,.2f}"


def _fmt_date(value: str, cfg: RenderConfig, rng: random.Random) -> str:
    """Render an ISO date in one of several real-world orders."""
    if not cfg.varied_dates or len(value) != 10 or value[4] != "-":
        return value
    year, month, day = value[:4], value[5:7], value[8:10]
    months = (
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    )
    try:
        month_name = months[int(month) - 1]
    except (ValueError, IndexError):
        return value
    return rng.choice(
        (f"{day}/{month}/{year}", f"{day} {month_name} {year}", f"{month}/{day}/{year}")
    )


def _label(canonical: str, cfg: RenderConfig, rng: random.Random) -> str:
    options = FIELD_ALIASES.get(canonical, (canonical,))
    return rng.choice(options) if cfg.use_label_synonyms else options[0]


def _prose(rendered: str, currency: str, rng: random.Random) -> str:
    """State a value in running prose, with no label to anchor on."""
    return rng.choice(
        (
            f"Funds should be sent to the account {rendered} held in London.",
            f"Settlement is to the account number {rendered}.",
            f"The total payable is {currency} {rendered}.",
            f"Kindly remit {rendered} in settlement of the balance.",
        )
    )


def render_document(
    values: dict[str, str],
    doc_id: str,
    difficulty: str = "clean",
    seed: int = 0,
) -> Document:
    """Render canonical values into a document, recording the exact truth.

    `truth` names only the fields actually placed in the text, so an extractor
    that invents a field the renderer omitted is scored wrong rather than merely
    uncounted.
    """
    cfg = DIFFICULTIES[difficulty]
    rng = random.Random(seed)
    truth: dict[str, str] = {}
    surface: dict[str, str] = {}
    notes: list[str] = []

    currency = values.get("Amt_InstdAmt_Ccy", "")
    amount_surface = ""
    if values.get("Amt_InstdAmt"):
        amount_surface = _fmt_amount(values["Amt_InstdAmt"], cfg)

    order = [k for k in FIELD_ALIASES if values.get(k)]
    rng.shuffle(order)

    lines: list[str] = []
    for key in order:
        raw = values[key]
        if not raw:
            continue
        if key == "Amt_InstdAmt":
            truth[key] = raw
            surface[key] = amount_surface
            # The currency rides along with the amount, as it does on paper.
            if currency:
                truth["Amt_InstdAmt_Ccy"] = currency
                surface["Amt_InstdAmt_Ccy"] = currency
            if key in cfg.bury_in_prose:
                notes.append(f"{key} stated in prose, no label")
                lines.append(_prose(amount_surface, currency, rng))
            else:
                shown = f"{currency} {amount_surface}".strip()
                lines.append(f"{_label(key, cfg, rng)}: {shown}")
            continue

        if key == "ReqdExctnDt_Dt":
            rendered = _fmt_date(raw, cfg, rng)
        else:
            rendered = raw
        truth[key] = raw
        surface[key] = rendered

        if key in cfg.bury_in_prose:
            notes.append(f"{key} stated in prose, no label")
            lines.append(_prose(rendered, currency, rng))
        elif key in cfg.omit_label:
            notes.append(f"{key} placed without a label")
            lines.append(rendered)
        else:
            lines.append(f"{_label(key, cfg, rng)}: {rendered}")

    for _ in range(cfg.distractors):
        label = rng.choice(DISTRACTOR_LABELS)
        noise = rng.choice(
            (
                f"{rng.randint(1000, 99999)}",
                f"{rng.randint(10, 99)}-{rng.randint(100000, 999999)}",
                f"+44 {rng.randint(1000000000, 9999999999)}",
                f"GB{rng.randint(10, 99)}BUKB{rng.randint(10000000, 99999999)}",
            )
        )
        lines.append(f"{label}: {noise}")

    rng.shuffle(lines)
    return Document(
        doc_id=doc_id,
        text="\n".join(lines),
        truth=truth,
        surface=surface,
        difficulty=difficulty,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# Real-value harvesting: pull canonical values out of the harvested fixtures.
# --------------------------------------------------------------------------


def values_from_mt(text: str) -> dict[str, str]:
    """Canonical renderable values from a real MT103 fixture.

    Routes through `mt103_to_dict` so the corpus speaks the same key space as
    the migration converter. A fixture that fails to parse yields {} and is
    skipped by the caller rather than silently contributing empty fields.
    """
    try:
        flat = mt103_to_dict(parse_mt103(text))
    except Exception:  # noqa: BLE001 - malformed fixture, skipped upstream
        return {}
    return _canonicalise(flat)


def values_from_mx(xml_text: str) -> dict[str, str]:
    """Canonical renderable values from a real MX fixture."""
    try:
        flat = mx_to_dict(xml_text)
    except Exception:  # noqa: BLE001 - malformed fixture, skipped upstream
        return {}
    return _canonicalise(flat)


# Message vocabulary -> document vocabulary. The creditor is the beneficiary on
# an outbound instruction, which is the direction a bank's own document describes.
_MESSAGE_TO_DOCUMENT: dict[str, str] = {
    "Cdtr_Nm": "Cdtr_Nm",
    "CdtrAcct_IBAN": "CdtrAcct_IBAN",
    "CdtrAgt_BICFI": "CdtrAgt_BICFI",
    "Dbtr_Nm": "Dbtr_Nm",
    "DbtrAcct_IBAN": "DbtrAcct_IBAN",
    "Amt_IntrBkSttlmAmt": "Amt_InstdAmt",
    "Amt_IntrBkSttlmAmt_Ccy": "Amt_InstdAmt_Ccy",
    "PmtId_EndToEndId": "PmtId_EndToEndId",
    "IntrBkSttlmDt": "ReqdExctnDt_Dt",
    "RmtInf_Ustrd": "RmtInf_Ustrd",
}


def _canonicalise(flat: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for message_key, value in flat.items():
        doc_key = _MESSAGE_TO_DOCUMENT.get(message_key)
        if doc_key is None or not value:
            continue
        out[doc_key] = str(value).strip()
    if "Amt_InstdAmt" in out:
        try:
            out["Amt_InstdAmt"] = f"{float(out['Amt_InstdAmt']):.2f}"
        except ValueError:
            del out["Amt_InstdAmt"]
    return out


@dataclass
class Corpus:
    documents: list[Document] = field(default_factory=list)

    def add(self, doc: Document) -> None:
        self.documents.append(doc)

    def by_difficulty(self, difficulty: str) -> list[Document]:
        return [d for d in self.documents if d.difficulty == difficulty]

    def field_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for doc in self.documents:
            for key in doc.truth:
                counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def summary(self) -> str:
        lines = [f"corpus: {len(self.documents)} documents"]
        for level in DIFFICULTIES:
            docs = self.by_difficulty(level)
            if not docs:
                continue
            avg = sum(len(d.truth) for d in docs) / len(docs)
            lines.append(f"  {level:<8} {len(docs):>5} docs, {avg:.1f} fields/doc")
        lines.append(f"  fields: {self.field_counts()}")
        return "\n".join(lines)


def build_corpus(
    seeds: list[dict[str, str]],
    per_seed: int = 1,
    seed: int = 0,
    one_level_per_seed: bool = False,
) -> Corpus:
    """Render every seed value-set once per difficulty and per repetition.

    ``one_level_per_seed`` gives each value-set a **single** difficulty, cycling
    through the levels, instead of rendering it at all three. The document count
    is unchanged; the number of distinct value-sets it contains is tripled.

    That matters because generalisation here is bounded by how many distinct
    value-sets the model sees, not by how many documents. Rendering all three
    levels means each value-set appears three times, which reinforces the
    document->JSON mapping for that value-set -- exactly the memorisation the
    held-out evaluation penalised -- while contributing no new values. Measured
    on a model trained the old way: 97.5% field recall on a held-out slice of its
    own value draw, 61.2% on a fresh one.

    Cycling levels rather than fixing one keeps the clean/messy/hostile coverage
    intact, because every third value-set gets each level.
    """
    rng = random.Random(seed)
    corpus = Corpus()
    counter = 0

    if one_level_per_seed:
        # DIFFICULTIES is a name -> RenderConfig mapping, so cycle its keys.
        levels = list(DIFFICULTIES)
        for index, values in enumerate(seeds):
            if not values:
                continue
            counter += 1
            doc = render_document(
                values,
                doc_id=f"doc{counter:05d}",
                difficulty=levels[index % len(levels)],
                seed=rng.randint(0, 1 << 30),
            )
            if doc.truth:
                corpus.add(doc)
        return corpus

    for difficulty in DIFFICULTIES:
        for _ in range(per_seed):
            for values in seeds:
                if not values:
                    continue
                counter += 1
                doc = render_document(
                    values,
                    doc_id=f"doc{counter:05d}",
                    difficulty=difficulty,
                    seed=rng.randint(0, 1 << 30),
                )
                if doc.truth:
                    corpus.add(doc)
    return corpus


def seeds_from_fixtures(limit: int = 200) -> list[dict[str, str]]:
    """Harvest real value-sets from the fixtures under data/."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "data"
    seeds: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "{1:" not in text and ":20:" not in text:
            continue
        values = values_from_mt(text)
        if len(values) >= 4:
            seeds.append(values)
    for path in sorted(root.rglob("*.xml")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        values = values_from_mx(text)
        if len(values) >= 4:
            seeds.append(values)
    return seeds[:limit]


def main() -> int:
    seeds = seeds_from_fixtures()
    print(f"harvested {len(seeds)} seed value-sets from real fixtures")
    if not seeds:
        return 1
    print(f"  example: {seeds[0]}")
    corpus = build_corpus(seeds, per_seed=1, seed=7)
    print()
    print(corpus.summary())
    print()
    for difficulty in DIFFICULTIES:
        docs = corpus.by_difficulty(difficulty)
        if not docs:
            continue
        doc = docs[0]
        print(f"--- {difficulty} ({doc.doc_id}) ---")
        print(doc.text)
        print(f"  truth  : {doc.truth}")
        print(f"  surface: {doc.surface}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
