"""Label the corpus with the real teacher, producing distillation targets.

This is the bridge between Phase 3 (measurement) and Phase 4 (the trained
student). It does three jobs at once:

1. **Produces the model column.** The same documents the rule baseline was
   scored on get labelled by DeepSeek Flash, so the two are directly
   comparable on STP and hallucination rate.

2. **Produces distillation targets.** The student is trained to imitate these
   labels, which is why the teacher has to be good rather than merely present.

3. **Separates supervision types.** The teacher supplies *coverage* -- which
   fields exist and where. Deterministic code supplies *precision*: canonical
   normalisation of amounts and dates, and checksum validation of IBANs and
   BICs. The teacher proved it needs this: on the first real call it returned
   `1,234.56`, the surface form, which is not a valid ISO 20022 amount. A
   student trained on that raw output would learn to copy commas.

Runs are resumable: results append to JSONL and already-labelled documents are
skipped, so an interrupted run costs nothing but the calls it already made.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from iso20022_lab.baseline import _norm_amount, _norm_date
from iso20022_lab.corpus import Corpus, Document
from iso20022_lab.distill import field_specs, json_schema_for
from iso20022_lab.validators import validate_field
from iso20022_lab.paths import data_path, schema_path
from iso20022_lab.xsd_introspect import XSDModel

DEFAULT_OUT = data_path("distill", "teacher_labels.jsonl")


@dataclass
class TeacherRecord:
    doc_id: str
    difficulty: str
    text: str
    truth: dict[str, str]
    teacher_raw: dict[str, str] = field(default_factory=dict)
    teacher_canonical: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


def canonicalise(values: dict[str, Any]) -> dict[str, str]:
    """Normalise teacher output into the canonical space the schema needs.

    This is where deterministic code buys precision the model cannot be trusted
    with: an amount is reformatted from whatever convention the teacher echoed,
    a date is parsed into ISO 8601, and an IBAN is checked before it is kept.
    """
    out: dict[str, str] = {}
    for key, raw in values.items():
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        if key == "Amt_InstdAmt":
            # The teacher often echoes "EUR 1,234.56" or "1,234.56".
            token = text.split()[-1]
            normalised = _norm_amount(token)
            if normalised:
                out[key] = normalised
            continue
        if key == "Amt_InstdAmt_Ccy":
            token = "".join(ch for ch in text if ch.isalpha()).upper()
            if len(token) == 3:
                out[key] = token
            continue
        if key == "ReqdExctnDt_Dt":
            normalised = _norm_date(text)
            if normalised:
                out[key] = normalised
            continue
        if key in {"CdtrAcct_IBAN", "DbtrAcct_IBAN"}:
            compact = text.replace(" ", "").upper()
            # `validate_field` only emits a Finding when a rule fails, so any
            # finding at all means the value is unusable as training data.
            if validate_field("IBAN", compact):
                continue  # a bad checksum is dropped, not propagated to training
            out[key] = compact
            continue
        if key.endswith("_BICFI"):
            compact = text.replace(" ", "").upper()
            if validate_field("BIC", compact):
                continue
            out[key] = compact
            continue
        out[key] = text
    return out


def load_done(path: Path) -> set[str]:
    """document ids already labelled, so a rerun resumes rather than repeats."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["doc_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def label_corpus(
    corpus: Corpus,
    model_path: str | Path,
    out_path: Path = DEFAULT_OUT,
    limit: int | None = None,
    sleep: float = 0.0,
    progress_every: int = 10,
) -> dict[str, int]:
    """Label every unlabelled document, appending to `out_path`."""
    from iso20022_lab.teacher import TeacherClient, cost_report

    model = XSDModel(Path(model_path))
    schema = json_schema_for(field_specs(model))
    client = TeacherClient()
    done = load_done(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pending = [d for d in corpus.documents if d.doc_id not in done]
    if limit is not None:
        pending = pending[:limit]

    stats = {"labelled": 0, "failed": 0, "skipped": len(done), "pending": len(pending)}
    if not pending:
        stats["cost"] = 0
        print(f"nothing to do: {len(done)} already labelled")
        return stats

    print(f"labelling {len(pending)} documents (skipping {len(done)} done)")
    with out_path.open("a", encoding="utf-8") as handle:
        for index, doc in enumerate(pending, 1):
            record = _label_one(client, doc, schema)
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            handle.flush()
            if record.ok:
                stats["labelled"] += 1
            else:
                stats["failed"] += 1
            if index % progress_every == 0 or index == len(pending):
                print(
                    f"  {index}/{len(pending)}  ok={stats['labelled']} "
                    f"failed={stats['failed']}  calls={client.calls}"
                )
            if sleep:
                time.sleep(sleep)

    print()
    print(cost_report(client))
    return stats


def _label_one(
    client: Any,
    doc: Document,
    schema: dict[str, Any],
) -> TeacherRecord:
    start = time.monotonic()
    try:
        label = client.label(doc.text, schema, temperature=0.0)
    except Exception as exc:  # noqa: BLE001 - one bad call must not kill the run
        return TeacherRecord(
            doc_id=doc.doc_id,
            difficulty=doc.difficulty,
            text=doc.text,
            truth=doc.truth,
            error=f"{type(exc).__name__}: {exc}",
            seconds=time.monotonic() - start,
        )
    raw = {k: str(v) for k, v in (label.values or {}).items()}
    return TeacherRecord(
        doc_id=doc.doc_id,
        difficulty=doc.difficulty,
        text=doc.text,
        truth=doc.truth,
        teacher_raw=raw,
        teacher_canonical=canonicalise(label.values or {}),
        error=label.error,
        seconds=time.monotonic() - start,
    )


def load_records(path: Path = DEFAULT_OUT) -> list[TeacherRecord]:
    if not path.exists():
        return []
    records: list[TeacherRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(
                TeacherRecord(
                    doc_id=payload["doc_id"],
                    difficulty=payload.get("difficulty", ""),
                    text=payload.get("text", ""),
                    truth=payload.get("truth", {}),
                    teacher_raw=payload.get("teacher_raw", {}),
                    teacher_canonical=payload.get("teacher_canonical", {}),
                    error=payload.get("error"),
                    seconds=payload.get("seconds", 0.0),
                )
            )
    return records


def main() -> int:
    import argparse

    from iso20022_lab.evaluate import build_measurement_corpus

    parser = argparse.ArgumentParser(description="label the corpus with the teacher")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--synth", type=int, default=60)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=str(schema_path()))
    args = parser.parse_args()

    corpus = build_measurement_corpus(synth_count=args.synth, seed=args.seed)
    print(f"corpus: {len(corpus.documents)} documents")
    stats = label_corpus(
        corpus,
        model_path=args.model,
        out_path=args.out,
        limit=args.limit,
    )
    print()
    print(f"result: {stats}")
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
