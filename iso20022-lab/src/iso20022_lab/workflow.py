"""End-to-end capture workflow: why a model, and what a human still does.

The question this answers is not "which extractor is more accurate". It is:

    **What does each approach prevent a human from having to do, and why can
    that not be done with a regex parser?**

So this runs the whole workflow on the same documents and reports the **human
action** each extractor leaves behind:

  * STP      -- nothing to do. Payment can be released.
  * VERIFY   -- the extraction is mostly right; an analyst checks/corrects the
                fields that differ.
  * KEY      -- extraction failed or produced something unusable; an analyst
                keys the payment from the document.

`VERIFY` is not the same as `KEY`, and collapsing them is the most common way a
business case for automation gets inflated. Checking a populated form against a
source document is a different act from transcribing it, and it takes a
different amount of time.

Three extractors are compared:

1. **rules_v1** -- regex written for the ONE canonical format. This is the honest
   starting point: rules are written against the format you actually receive
   most of, and they work.
2. **rules_v2** -- rules extended with every label synonym in the vocabulary, plus
   the normalisers. This is the "just add more rules" response, and it is given
   its best shot.
3. **model** -- the trained model.

Running the first two is the whole point. Asserting "regex cannot do this" is
worthless; running both and showing the specific documents where v2 still fails,
and naming the class of failure, is the argument.

Every predicted extraction is also pushed through the real message builder and
the real XSD, because an extraction that cannot become a valid payment has not
solved the workflow.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from iso20022_lab.baseline import Extraction, RuleExtractor, _norm_amount, _norm_date
from iso20022_lab.corpus import FIELD_ALIASES, Document
from iso20022_lab.paths import data_path, schema_path

CANONICAL_LABELS: dict[str, str] = {k: v[0] for k, v in FIELD_ALIASES.items()}


# --------------------------------------------------------------------------
# Extractor 1: rules written for one known format
# --------------------------------------------------------------------------


class CanonicalRuleExtractor:
    """Regex for the single format the rules were written against.

    This is what a shop actually has before it starts adapting: a template
    matcher for the document layout that dominates its inbound. It is not a straw
    man -- on that layout it is correct every time, and it is faster, cheaper,
    deterministic and auditable. Everything below is about what happens outside
    that layout.
    """

    def __init__(self, labels: dict[str, str] | None = None):
        self.labels = labels or CANONICAL_LABELS
        self.patterns = {
            key: re.compile(
                rf"^\s*{re.escape(label)}\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
            )
            for key, label in self.labels.items()
        }

    def extract(self, text: str) -> Extraction:
        found = Extraction()
        for key, pattern in self.patterns.items():
            if key == "Amt_InstdAmt_Ccy":
                continue  # handled with the amount below
            m = pattern.search(text)
            if not m:
                continue
            value = _normalise(key, m.group(1).strip())
            if value:
                found.values[key] = value
                found.method[key] = "canonical-label"

        # The currency is not its own field on a real instruction; it rides with
        # the amount. An earlier version looked for a "Currency:" label that no
        # document contains, so it scored 0% STP on the very format it was written
        # for. That was a straw man: rules are written against the documents a shop
        # actually receives, so this reads the currency off the amount line.
        m = self.patterns.get("Amt_InstdAmt")
        if m:
            match = m.search(text)
            if match:
                ccy = _normalise("Amt_InstdAmt_Ccy", match.group(1).strip())
                if ccy:
                    found.values["Amt_InstdAmt_Ccy"] = ccy
                    found.method["Amt_InstdAmt_Ccy"] = "canonical-label-inline"
        return found


def _normalise(key: str, raw: str) -> str:
    """Canonicalise a captured value. Shared by both rule extractors."""
    if key == "Amt_InstdAmt":
        token = raw.split()[-1] if raw.split() else raw
        return _norm_amount(token)
    if key == "Amt_InstdAmt_Ccy":
        for token in re.split(r"[\s,]+", raw):
            if len(token) == 3 and token.isalpha() and token.isupper():
                return token
        return ""
    if key == "ReqdExctnDt_Dt":
        return _norm_date(raw)
    if key in {"CdtrAcct_IBAN", "DbtrAcct_IBAN"}:
        compact = re.sub(r"\s+", "", raw).upper()
        m = re.search(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", compact)
        return m.group(0) if m else ""
    if key == "PmtId_EndToEndId":
        return raw.split()[0].strip(",;") if raw.split() else ""
    return raw


# --------------------------------------------------------------------------
# Outcome classification and the human action it implies
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Extractor 3: the trained model
# --------------------------------------------------------------------------


class ModelExtractor:
    """The trained student, behind the same `extract` interface as the rules.

    This exists so the model is scored on exactly the same documents, by exactly
    the same code path, as the two rule engines above. A comparison run through
    a different harness is not a comparison.

    Loading is lazy and torch is imported in the constructor rather than at
    module scope, so the rules-only path never needs torch installed.
    """

    def __init__(
        self,
        model_dir: Path,
        tokenizer_dir: Path,
        device: str | None = None,
        max_new_tokens: int = 200,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # These are Any on purpose, not as a suppression. `trust_remote_code=True`
        # means the concrete classes come from the Hub at runtime, so their shape
        # is genuinely unknown to a type checker. The alternative -- naming a
        # concrete class -- would be a lie that breaks the moment the artifact is
        # loaded from somewhere else.
        self._torch: Any = torch
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            str(tokenizer_dir), trust_remote_code=True
        )
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            str(model_dir), trust_remote_code=True
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device != "cpu":
            torch.nn.Module.cuda(self.model)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def extract(self, text: str) -> Extraction:
        from iso20022_lab.model.evaluate_model import parse_generation
        from iso20022_lab.model.tokenizer import prompt_for

        found = Extraction()
        ids = self.tokenizer.encode(prompt_for(text), add_special_tokens=False)
        input_ids = self._torch.tensor([ids], device=self.device)
        with self._torch.no_grad():
            out = self.model.generate_greedy(input_ids, max_new_tokens=self.max_new_tokens)
        raw = self.tokenizer.decode(out[0, len(ids) :].tolist())
        for key, value in parse_generation(raw).items():
            found.values[key] = value
            found.method[key] = "model"
        return found


@dataclass
class Outcome:
    doc_id: str
    difficulty: str
    extractor: str
    action: str  # STP | VERIFY | KEY
    correct: int
    expected: int
    wrong: list[str]
    invented: list[str]
    xsd_valid: bool
    xsd_error: str = ""
    seconds: float = 0.0


def classify(
    truth: dict[str, str],
    extracted: dict[str, str],
    xsd_valid: bool,
) -> tuple[str, list[str], list[str]]:
    """Return (action, wrong_fields, invented_fields).

    The thresholds are judgement calls and are stated rather than buried:

    * No wrong or invented field, and the message validates -> STP.
    * At least 60% of fields right, no invented field, message validates
      -> VERIFY. The analyst is correcting a populated form.
    * Otherwise -> KEY. Either too little was recovered to be worth checking, or
      the extractor invented something, which means every field has to be
      re-established because the form can no longer be trusted.

    An invented field forces KEY on purpose. A wrong value that is obviously
    wrong gets caught in review; a plausible value that was never in the
    document is exactly what a reviewer waves through.
    """
    wrong = [k for k, v in truth.items() if extracted.get(k) != v]
    invented = [k for k, v in extracted.items() if v and k not in truth]
    correct = len(truth) - len(wrong)
    ratio = correct / len(truth) if truth else 0.0
    if not wrong and not invented and xsd_valid:
        return "STP", wrong, invented
    if invented or not xsd_valid or ratio < 0.6:
        return "KEY", wrong, invented
    return "VERIFY", wrong, invented


# --------------------------------------------------------------------------
# The workflow runner
# --------------------------------------------------------------------------


@dataclass
class WorkflowReport:
    extractor: str
    outcomes: list[Outcome] = field(default_factory=list)

    @property
    def actions(self) -> dict[str, int]:
        counts: dict[str, int] = {"STP": 0, "VERIFY": 0, "KEY": 0}
        for o in self.outcomes:
            counts[o.action] = counts.get(o.action, 0) + 1
        return counts

    @property
    def field_accuracy(self) -> float:
        total = sum(o.expected for o in self.outcomes)
        return sum(o.correct for o in self.outcomes) / total if total else 0.0

    @property
    def invented_per_doc(self) -> float:
        return (
            sum(len(o.invented) for o in self.outcomes) / len(self.outcomes)
            if self.outcomes
            else 0.0
        )

    @property
    def xsd_pass_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(1 for o in self.outcomes if o.xsd_valid) / len(self.outcomes)

    def by_difficulty(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for level in ("clean", "messy", "hostile"):
            rows = [o for o in self.outcomes if o.difficulty == level]
            if not rows:
                continue
            stp = sum(1 for o in rows if o.action == "STP")
            out[level] = {
                "docs": len(rows),
                "stp": stp / len(rows),
                "verify": sum(1 for o in rows if o.action == "VERIFY") / len(rows),
                "key": sum(1 for o in rows if o.action == "KEY") / len(rows),
            }
        return out

    def failure_modes(self) -> dict[str, int]:
        """Count wrong/missing fields by name, to name the class of failure."""
        counts: dict[str, int] = {}
        for o in self.outcomes:
            for k in o.wrong:
                counts[k] = counts.get(k, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def run_workflow(
    documents: list[Document],
    extractor: Any,
    extractor_name: str,
    xsd_path: str | Path,
    use_message_builder: bool = True,
) -> WorkflowReport:
    """Run documents through extraction, message build and schema validation."""
    from iso20022_lab.distill import to_slots
    from iso20022_lab.serialize import MessageBuilder, fill_system_fields
    from iso20022_lab.xsd_introspect import XSDModel

    report = WorkflowReport(extractor=extractor_name)
    model = XSDModel(Path(xsd_path)) if use_message_builder else None

    for doc in documents:
        started = time.monotonic()
        extracted = extractor.extract(doc.text)
        values = dict(extracted.values)

        xsd_valid = False
        xsd_error = ""
        if model is not None:
            try:
                slots = to_slots(values, model)
                fill_system_fields(slots, model)
                built = MessageBuilder(model).build(slots)
                xsd_valid = built.ok
                if not built.ok:
                    xsd_error = built.errors[0] if built.errors else "build failed"
            except Exception as exc:  # noqa: BLE001 - a crash is a KEY outcome
                xsd_error = f"{type(exc).__name__}: {exc}"

        action, wrong, invented = classify(doc.truth, values, xsd_valid)
        report.outcomes.append(
            Outcome(
                doc_id=doc.doc_id,
                difficulty=doc.difficulty,
                extractor=extractor_name,
                action=action,
                correct=len(doc.truth) - len(wrong),
                expected=len(doc.truth),
                wrong=wrong,
                invented=invented,
                xsd_valid=xsd_valid,
                xsd_error=xsd_error,
                seconds=time.monotonic() - started,
            )
        )
    return report


def render(reports: list[WorkflowReport]) -> str:
    lines: list[str] = []
    lines.append("=" * 88)
    lines.append("CAPTURE WORKFLOW: what each extractor leaves for a human to do")
    lines.append("=" * 88)
    lines.append(
        f"{'extractor':<14} {'docs':>5} {'STP':>7} {'VERIFY':>8} {'KEY':>7} "
        f"{'field%':>7} {'invented/doc':>13} {'XSD pass':>9}"
    )
    for r in reports:
        a = r.actions
        n = len(r.outcomes)
        lines.append(
            f"{r.extractor:<14} {n:>5} {a['STP'] / n * 100:>6.1f}% "
            f"{a['VERIFY'] / n * 100:>7.1f}% {a['KEY'] / n * 100:>6.1f}% "
            f"{r.field_accuracy * 100:>6.1f}% {r.invented_per_doc:>13.2f} "
            f"{r.xsd_pass_rate * 100:>8.1f}%"
        )
    lines.append("")
    for r in reports:
        lines.append(f"--- {r.extractor} by difficulty ---")
        for level, row in r.by_difficulty().items():
            lines.append(
                f"  {level:<9} n={int(row['docs']):>4}  STP {row['stp'] * 100:>5.1f}%  "
                f"VERIFY {row['verify'] * 100:>5.1f}%  KEY {row['key'] * 100:>5.1f}%"
            )
    lines.append("")
    for r in reports:
        modes = r.failure_modes()
        if modes:
            top = ", ".join(f"{k} ({v})" for k, v in list(modes.items())[:6])
            lines.append(f"--- {r.extractor}: most-missed fields ---")
            lines.append(f"  {top}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Human effort model - the thing the economics is actually about
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EffortModel:
    """Minutes of analyst time per outcome. These are the economic inputs."""

    minutes_key: float = 8.0
    minutes_verify: float = 1.5
    minutes_stp: float = 0.0
    loaded_hourly_cost: float = 45.0

    def minutes_per_document(self, actions: dict[str, int]) -> float:
        total = sum(actions.values()) or 1
        return (
            actions.get("KEY", 0) * self.minutes_key
            + actions.get("VERIFY", 0) * self.minutes_verify
            + actions.get("STP", 0) * self.minutes_stp
        ) / total


def effort_table(reports: list[WorkflowReport], effort: EffortModel) -> str:
    """Human effort and cost per document, per extractor."""
    lines: list[str] = []
    lines.append("=" * 88)
    lines.append("HUMAN EFFORT (the economics: inference cost is not the question)")
    lines.append("=" * 88)
    lines.append(
        f"assumptions: KEY {effort.minutes_key:.0f} min, "
        f"VERIFY {effort.minutes_verify:.1f} min, STP {effort.minutes_stp:.0f} min, "
        f"${effort.loaded_hourly_cost:.0f}/hour loaded"
    )
    lines.append("")
    lines.append(
        f"{'extractor':<14} {'min/doc':>9} {'vs KEY-all':>11} {'cost/1k docs':>13}"
    )
    baseline = effort.minutes_key
    for r in reports:
        per_doc = effort.minutes_per_document(r.actions)
        saved = (baseline - per_doc) / baseline * 100
        cost_1k = per_doc * 1000 / 60 * effort.loaded_hourly_cost
        lines.append(
            f"{r.extractor:<14} {per_doc:>8.2f} {saved:>10.1f}% ${cost_1k:>12,.0f}"
        )
    return "\n".join(lines)


def main() -> int:
    import argparse

    from iso20022_lab.corpus import build_corpus
    from iso20022_lab.synth import generate_many

    parser = argparse.ArgumentParser(description="capture workflow comparison")
    parser.add_argument("--docs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--model", default=None, help="trained model dir, optional")
    parser.add_argument(
        "--tokenizer",
        default=str(data_path("tokenizer")),
        help="tokenizer dir, used with --model",
    )
    parser.add_argument(
        "--xsd",
        default=str(schema_path()),
    )
    parser.add_argument("--out", type=Path, default=data_path("workflow.json"))
    args = parser.parse_args()

    corpus = build_corpus(
        generate_many(args.docs, seed=args.seed), per_seed=1, seed=args.seed
    )
    documents = corpus.documents
    print(f"documents: {len(documents)}")
    print()

    reports: list[WorkflowReport] = []
    reports.append(run_workflow(documents, CanonicalRuleExtractor(), "rules_v1", args.xsd))
    reports.append(run_workflow(documents, RuleExtractor(), "rules_v2", args.xsd))

    if args.model and Path(args.model).exists():
        reports.append(
            run_workflow(
                documents,
                ModelExtractor(Path(args.model), Path(args.tokenizer)),
                "model",
                args.xsd,
            )
        )
    elif args.model:
        print(f"--model {args.model} does not exist; skipping the model column")
        print()
    print(render(reports))
    print()
    print(effort_table(reports, EffortModel()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "documents": len(documents),
                "reports": [
                    {
                        "extractor": r.extractor,
                        "actions": r.actions,
                        "field_accuracy": r.field_accuracy,
                        "invented_per_doc": r.invented_per_doc,
                        "xsd_pass_rate": r.xsd_pass_rate,
                        "by_difficulty": r.by_difficulty(),
                        "failure_modes": r.failure_modes(),
                    }
                    for r in reports
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
