"""The Phase 3 measurement: where a rules engine actually plateaus.

PROCESS.md 3.3 says: "It needs a real number from a real process: exceptions per
day, minutes per exception, loaded cost per hour. Until that exists, treat it as
a hypothesis to validate in Phase 3, not a premise."

This module produces the exception rate. `economics.py` turns it into money.

Two levels of measurement, because they answer different questions:

* **Field accuracy** -- which fields the extractor gets right. Diagnostic.
* **Document STP rate** -- the fraction of documents where EVERY field is
  correct. This is the business metric, and it is the one that matters, because
  a payment is not "94% correct". One wrong field is one exception, so a per-field
  average systematically overstates straight-through performance. The gap between
  the two numbers is the whole point of measuring both.

A spurious field is counted as a failure even when every expected field is right.
An extractor that invents a payer name is producing a message that will be
wrong, and scoring it as a pass would hide exactly the failure mode this
exercise exists to find.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from iso20022_lab.baseline import Extraction, RuleExtractor
from iso20022_lab.corpus import DIFFICULTIES, Corpus, Document


@dataclass
class FieldScore:
    key: str
    expected: int = 0
    correct: int = 0
    spurious: int = 0

    @property
    def recall(self) -> float:
        return self.correct / self.expected if self.expected else 0.0

    @property
    def precision(self) -> float:
        denom = self.correct + self.spurious
        return self.correct / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class DifficultyReport:
    difficulty: str
    docs: int = 0
    stp_docs: int = 0
    expected: int = 0
    correct: int = 0
    spurious: int = 0
    fields: dict[str, FieldScore] = field(default_factory=dict)
    # Documents that failed, with the reason, for inspection.
    failures: list[tuple[str, list[str]]] = field(default_factory=list)

    @property
    def stp_rate(self) -> float:
        return self.stp_docs / self.docs if self.docs else 0.0

    @property
    def exception_rate(self) -> float:
        return 1.0 - self.stp_rate

    @property
    def field_accuracy(self) -> float:
        return self.correct / self.expected if self.expected else 0.0

    @property
    def hallucination_rate(self) -> float:
        """Spurious fields per document: values invented that were not present."""
        return self.spurious / self.docs if self.docs else 0.0


@dataclass
class Report:
    extractor: str
    by_difficulty: dict[str, DifficultyReport] = field(default_factory=dict)

    @property
    def overall(self) -> DifficultyReport:
        merged = DifficultyReport(difficulty="all")
        for report in self.by_difficulty.values():
            merged.docs += report.docs
            merged.stp_docs += report.stp_docs
            merged.expected += report.expected
            merged.correct += report.correct
            merged.spurious += report.spurious
            for key, score in report.fields.items():
                acc = merged.fields.setdefault(key, FieldScore(key=key))
                acc.expected += score.expected
                acc.correct += score.correct
                acc.spurious += score.spurious
            merged.failures.extend(report.failures)
        return merged

    def render(self) -> str:
        lines: list[str] = []
        lines.append("=" * 78)
        lines.append(f"EXTRACTOR: {self.extractor}")
        lines.append("=" * 78)
        lines.append(
            f"{'difficulty':<10} {'docs':>6} {'STP%':>7} {'except%':>8} "
            f"{'field%':>8} {'invented/doc':>13}"
        )
        for name in DIFFICULTIES:
            report = self.by_difficulty.get(name)
            if report is None or not report.docs:
                continue
            lines.append(
                f"{name:<10} {report.docs:>6} {report.stp_rate * 100:>6.1f}% "
                f"{report.exception_rate * 100:>7.1f}% "
                f"{report.field_accuracy * 100:>7.1f}% "
                f"{report.hallucination_rate:>13.2f}"
            )
        overall = self.overall
        lines.append("-" * 78)
        lines.append(
            f"{'ALL':<10} {overall.docs:>6} {overall.stp_rate * 100:>6.1f}% "
            f"{overall.exception_rate * 100:>7.1f}% "
            f"{overall.field_accuracy * 100:>7.1f}% "
            f"{overall.hallucination_rate:>13.2f}"
        )
        lines.append("")
        gap = overall.field_accuracy - overall.stp_rate
        lines.append(
            f"  field accuracy {overall.field_accuracy * 100:.1f}% vs "
            f"STP {overall.stp_rate * 100:.1f}%  ->  gap {gap * 100:.1f} points"
        )
        lines.append("  The gap is the whole finding: a per-field average overstates")
        lines.append(
            "  straight-through performance, because one wrong field is one exception."
        )
        lines.append("")
        lines.append("  per-field breakdown:")
        lines.append(f"    {'field':<22} {'exp':>5} {'ok':>5} {'recall':>7} {'prec':>7}")
        for key, score in sorted(overall.fields.items(), key=lambda kv: kv[1].recall):
            lines.append(
                f"    {key:<22} {score.expected:>5} {score.correct:>5} "
                f"{score.recall * 100:>6.1f}% {score.precision * 100:>6.1f}%"
            )
        return "\n".join(lines)


def score_document(doc: Document, found: Extraction) -> tuple[list[str], list[str]]:
    """Return (wrong_or_missing, spurious) for one document."""
    wrong: list[str] = []
    for key, want in doc.truth.items():
        got = found.get(key)
        if got != want:
            wrong.append(key)
    spurious = [k for k, v in found.values.items() if v and k not in doc.truth]
    return wrong, spurious


def evaluate(
    corpus: Corpus,
    extractor: RuleExtractor | None = None,
    name: str = "rule baseline",
) -> Report:
    """Score an extractor across the corpus, per difficulty level."""
    extractor = extractor or RuleExtractor()
    report = Report(extractor=name)

    for doc in corpus.documents:
        level = report.by_difficulty.setdefault(
            doc.difficulty, DifficultyReport(difficulty=doc.difficulty)
        )
        level.docs += 1
        found = extractor.extract(doc.text)

        for key, want in doc.truth.items():
            score = level.fields.setdefault(key, FieldScore(key=key))
            score.expected += 1
            level.expected += 1
            if found.get(key) == want:
                score.correct += 1
                level.correct += 1

        spurious = [k for k, v in found.values.items() if v and k not in doc.truth]
        for key in spurious:
            score = level.fields.setdefault(key, FieldScore(key=key))
            score.spurious += 1
            level.spurious += 1

        wrong, _ = score_document(doc, found)
        if not wrong and not spurious:
            level.stp_docs += 1
        else:
            reasons = [f"wrong:{k}" for k in wrong] + [f"invented:{k}" for k in spurious]
            level.failures.append((doc.doc_id, reasons))

    return report


def failure_examples(report: Report, difficulty: str, limit: int = 5) -> str:
    """Render a few concrete failures, since the reasons matter more than the rate."""
    level = report.by_difficulty.get(difficulty)
    if level is None or not level.failures:
        return f"(no failures at {difficulty})"
    lines = [
        f"--- {difficulty}: {len(level.failures)} failing documents, first {limit} ---"
    ]
    for doc_id, reasons in level.failures[:limit]:
        lines.append(f"  {doc_id}: {', '.join(reasons)}")
    return "\n".join(lines)


def build_measurement_corpus(
    synth_count: int = 60,
    seed: int = 0,
) -> Corpus:
    """Real harvested seeds plus validated synthetic values, across difficulties.

    Both halves are labelled in the report so synthetic volume is never mistaken
    for real volume.
    """
    from iso20022_lab.corpus import build_corpus, seeds_from_fixtures
    from iso20022_lab.synth import generate_many

    real = seeds_from_fixtures()
    synthetic = generate_many(synth_count, seed=seed)
    return build_corpus(real + synthetic, per_seed=1, seed=seed)


def main() -> int:
    from iso20022_lab.baseline import status as baseline_status
    from iso20022_lab.synth import main as synth_main

    print(baseline_status())
    print()
    if synth_main() != 0:
        print("value synthesis produced invalid values; aborting")
        return 1
    print()

    corpus = build_measurement_corpus(synth_count=60, seed=13)
    print(corpus.summary())
    print()

    report = evaluate(corpus)
    print(report.render())
    print()
    for difficulty in DIFFICULTIES:
        print(failure_examples(report, difficulty, limit=4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
