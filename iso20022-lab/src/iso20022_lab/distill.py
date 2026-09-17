"""
Teacher -> student distillation for ISO 20022 extraction.

Why distillation rather than plain supervision: producing a labelled
(document, correct ISO 20022 message) pair by hand is the expensive part of this
whole project, and section 7 of PROCESS.md established that the paired data
simply does not exist publicly. A teacher converts that recurring human-labelling
cost into a one-time API cost, and the student is then free forever.

The constraint that shapes the design: DeepSeek V4 Flash is a text API, so it
returns sequences, not logits. Classical soft-target distillation is unavailable.
What this pipeline does instead is **filtered sequence-level distillation**
(Kim & Rush 2016, arXiv 1606.07947):

    teacher supplies COVERAGE       -- many candidate labels, cheaply
    validators supply PRECISION     -- every label is checked mechanically

That inversion is what lets a cheap teacher be good enough. A teacher that is
right 70% of the time is useless unfiltered and valuable filtered, because the
filters are deterministic and auditable.

Three gates stand between a teacher label and the training set:

    1. GROUNDED    every value must occur in the source span
    2. VALIDATED   IBAN mod-97, BIC, UETR, LEI, ISO 4217 minor units
    3. SERIALIZABLE builds a message that validates against the real XSD

Anything failing a gate is dropped and counted, so teacher quality is measurable
rather than assumed.

Run the self-test with no API key and no spend:

    python iso20022-lab/src/iso20022_lab/distill.py --selftest
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from iso20022_lab.serialize import (
    MessageBuilder,
    fill_system_fields,
    validate,
)
from iso20022_lab.teacher import TeacherClient, TeacherLabel, cost_report
from iso20022_lab.validators import amount_matches_minor_units, validate_field
from iso20022_lab.xsd_introspect import Element, XSDModel

# __file__ = <repo>/iso20022-lab/src/iso20022_lab/distill.py
#   parents[2] = <repo>/iso20022-lab   -> holds schemas/
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = PACKAGE_ROOT / "schemas"


# --------------------------------------------------------------------- schema

# Fields the *system* supplies when building a message. A source document never
# contains a value for these, so asking a teacher to extract them guarantees an
# ungrounded answer. MsgId and CreDtTm are generated; NbOfTxs and CtrlSum are
# derived from the transaction list.
GENERATED_FIELDS = frozenset(
    {"MsgId", "CreDtTm", "NbOfTxs", "CtrlSum", "BtchBookg", "Grpg"}
)

# Branches that are irrelevant for a corporate payment document. PrvtId and
# DtAndPlcOfBirth describe a private individual's birth details, which is a
# different identification branch of the same PartyChoice and never appears on an
# invoice; AdrTp/Prtry is an address-type qualifier, not an address.
EXCLUDED_MARKERS = (
    "PrvtId/",
    "DtAndPlcOfBirth/",
    "AdrTp/",
    "FwdgAgt/",
    "Chrtry/",
    "/Prtry/",
)

# Element names worth extracting, ordered by how often a real payment document
# carries them. Order matters: it is the priority for filling the field set.
#
# Note what is NOT here as a filter: `required`. For extraction, requiring
# minOccurs=1 selects exactly the wrong fields. `Nm` is optional because a party
# may be identified by account instead, and `IBAN` is a branch of an xs:choice.
# Both are precisely what a document does contain.
FIELD_PRIORITY = (
    "Nm",
    "IBAN",
    "BICFI",
    "InstdAmt",
    "IntrBkSttlmAmt",
    "EndToEndId",
    "InstrId",
    "TxId",
    "UETR",
    "IntrBkSttlmDt",
    "ReqdExctnDt",
    "Dt",
    "DtTm",
    "ChrgBr",
    "Purp",
    "CtgyPurp",
    "Ustrd",
)

# Branches that produce noise rather than signal. `Othr` and `Prtry` are the
# proprietary escape hatches of a choice (captured only when nothing standard
# fits); `Prxy` is a proxy identifier; `ChqInstr` is cheque handling detail.
NOISE_MARKERS = (
    "/Othr/",
    "/Prtry/",
    "/Prxy/",
    "ChqInstr/",
    "PrvtId/",
    "DtAndPlcOfBirth/",
    "AdrTp/",
    "FwdgAgt/",
)

# Container names that carry no identifying information in a key.
BOILERPLATE = frozenset(
    {"Document", "CstmrCdtTrfInitn", "PmtInf", "CdtTrfTxInf", "Id", "Tp"}
)

# The roles that actually appear on a commercial payment document. Without this
# filter the field set fills up with Grnshee, Invcr and BrnchId, which are valid
# ISO 20022 and never present in the documents this system reads.
CORE_ROLES = frozenset(
    {
        "Dbtr",
        "Cdtr",
        "UltmtDbtr",
        "UltmtCdtr",
        "InitgPty",
        "DbtrAcct",
        "CdtrAcct",
        "DbtrAgt",
        "CdtrAgt",
        "InstdAmt",
        "IntrBkSttlmAmt",
        "PmtId",
        "GrpHdr",
        "ReqdExctnDt",
    }
)


def _role(path: str) -> str | None:
    """The core payment role a path belongs to, or None if it is not one."""
    for part in path.split("/"):
        if part in CORE_ROLES:
            return part
    return None


@dataclass
class FieldSpec:
    """One extractable field, derived from the schema rather than hand-written."""

    key: str  # Cdtr_Nm, CdtrAcct_IBAN, InstdAmt_Ccy ...
    path: str  # CstmrCdtTrfInitn/PmtInf/CdtTrfTxInf/Cdtr/Nm
    name: str  # element or attribute name
    is_attribute: bool = False
    code_list: tuple[str, ...] = ()


def field_key(path: str, extra: str = "") -> str:
    """Stable, unique, readable key for a schema path.

    Element names alone are not unique -- `Id` occurs four times in the first
    fourteen fields of pain.001 -- so keys carry path context or values get
    silently misplaced.
    """
    parts = [p for p in path.split("/") if p and p not in BOILERPLATE]
    tail = parts[-2:] if len(parts) >= 2 else parts
    key = "_".join(tail)
    return f"{key}_{extra}" if extra else key


def field_specs(model: XSDModel, limit: int = 24) -> list[FieldSpec]:
    """Derive the extractable field set from the XSD.

    Three things this deliberately does that a naive selection gets wrong:

    1. It includes OPTIONAL elements and xs:choice branches. `Nm` is optional
       because a party may be identified by account instead, and `IBAN` is a
       branch of a choice -- yet both are exactly what a document carries.
       Filtering on minOccurs=1 selects the wrong field set.
    2. It draws candidates in round-robin order across field TYPES. Scanning name
       by name lets the many `Nm` variants consume the whole budget before `IBAN`
       is ever considered.
    3. It restricts to core payment roles, so the set is what a payment document
       contains rather than everything the schema permits.
    """
    leaves = [(p, el) for p, el in model.paths() if el.is_leaf]

    # name -> candidate (path, element) pairs, in schema order
    buckets: dict[str, list[tuple[str, Element]]] = {name: [] for name in FIELD_PRIORITY}
    for path, el in leaves:
        if el.name not in buckets:
            continue
        if _role(path) is None:
            continue
        if any(marker in path for marker in NOISE_MARKERS):
            continue
        buckets[el.name].append((path, el))

    specs: list[FieldSpec] = []
    seen: set[str] = set()

    def add(spec: FieldSpec) -> None:
        if spec.key not in seen:
            seen.add(spec.key)
            specs.append(spec)

    # Round-robin across field types so the set is balanced.
    while len(specs) < limit and any(buckets.values()):
        for name in FIELD_PRIORITY:
            if not buckets[name]:
                continue
            path, el = buckets[name].pop(0)
            add(FieldSpec(field_key(path), path, name, code_list=el.enumerations))
            # An amount carries Ccy as an XML attribute, so it never appears as a
            # leaf. Surface it as its own field or the teacher cannot supply it.
            if "Ccy" in el.attribute_names:
                add(FieldSpec(field_key(path, "Ccy"), path, "Ccy", is_attribute=True))
            if len(specs) >= limit:
                break
    return specs


def json_schema_for(specs: list[FieldSpec]) -> dict[str, Any]:
    """A compact JSON Schema handed to the teacher as its output contract."""
    properties: dict[str, Any] = {}
    for spec in specs:
        described = f"{spec.path}/{spec.name}" if spec.is_attribute else spec.path
        entry: dict[str, Any] = {"type": "string", "description": described}
        if spec.code_list:
            entry["enum"] = list(spec.code_list)
        properties[spec.key] = entry
    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }


# ------------------------------------------------------------------- filtering


@dataclass
class FilterStats:
    seen: int = 0
    grounded: int = 0
    validated: int = 0
    serializable: int = 0
    kept: int = 0
    reasons: Counter[str] = field(default_factory=Counter)

    def row(self) -> str:
        return (
            f"seen={self.seen} grounded={self.grounded} validated={self.validated} "
            f"serializable={self.serializable} kept={self.kept}"
        )


def grounded(values: dict[str, Any], span: str) -> list[str]:
    """Every value must appear in the span. Case- and whitespace-insensitive.

    This is the check that catches teacher hallucination, and it is the same
    principle the runtime grounding check enforces.
    """
    haystack = re.sub(r"\s+", " ", span).lower()
    missing: list[str] = []
    for name, value in values.items():
        needle = re.sub(r"\s+", " ", str(value)).lower().strip()
        if needle and needle not in haystack:
            missing.append(name)
    return missing


def validated(values: dict[str, Any]) -> list[str]:
    """Deterministic validator findings across all fields."""
    problems: list[str] = []
    for name, value in values.items():
        problems.extend(str(f) for f in validate_field(name, value))
    # amount/currency consistency, when both are present
    amount = next(
        (v for k, v in values.items() if k in {"Amt", "InstdAmt", "TtlAmt"}), None
    )
    ccy = next((v for k, v in values.items() if "Ccy" in k or k == "currency"), None)
    if amount is not None and ccy is not None:
        if not amount_matches_minor_units(str(amount), str(ccy)):
            problems.append(
                f"amount {amount!r} does not match {ccy} minor units "
                f"(PROCESS.md section 2.8)"
            )
    return problems


def to_slots(values: dict[str, Any], model: XSDModel) -> dict[str, Any]:
    """Map teacher values onto the nested structure the serializer wants.

    Keys are path-derived and therefore exact: a value for `Cdtr_Nm` lands under
    Cdtr/Nm because the key encodes that path, not because a name happened to
    match. Currency attributes are folded into their amount as `Ccy`, which is
    the shape the serializer expects for a simpleContent amount element.
    """
    root_name = model.message().children[0].name
    slots: dict[str, Any] = {root_name: {}}
    specs = {s.key: s for s in field_specs(model)}

    def container_for(parts: list[str]) -> dict[str, Any]:
        node = slots[root_name]
        for part in parts:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        return node

    deferred_ccy: list[tuple[list[str], Any]] = []
    for key, value in values.items():
        spec = specs.get(key)
        if spec is None:
            continue  # unknown key: dropped, never silently placed
        # Drop BOTH `Document` and the root message element. `slots[root_name]`
        # already represents the root, so keeping the root name in the parts
        # nests CstmrCdtTrfInitn inside itself and every value lands one level
        # too deep -- which then looks like "required field missing".
        parts = spec.path.split("/")[2:]
        if not parts:
            continue
        if spec.is_attribute:
            deferred_ccy.append((parts, value))
            continue
        container_for(parts[:-1])[parts[-1]] = value

    # Amount elements need {"_value": ..., "Ccy": ...} to carry the attribute.
    # The scalar amount must be MOVED into `_value`, not replaced: descending with
    # container_for and assigning straight onto the element overwrites the amount
    # with an empty mapping, silently producing a message with no value in it.
    for parts, ccy in deferred_ccy:
        parent = container_for(parts[:-1])
        name = parts[-1]
        current = parent.get(name)
        if isinstance(current, dict):
            current["Ccy"] = ccy
        elif current is not None:
            parent[name] = {"_value": current, "Ccy": ccy}
        else:
            parent[name] = {"Ccy": ccy}
    return slots


@dataclass
class DistilledExample:
    span: str
    values: dict[str, Any]
    reasoning: str
    agreement: float
    xml: str | None = None


def distill_one(
    span: str,
    specs: list[FieldSpec],
    model: XSDModel,
    label: TeacherLabel,
    agreement: float = 1.0,
    stats: FilterStats | None = None,
) -> DistilledExample | None:
    """Run one teacher label through the three gates."""
    stats = stats or FilterStats()
    stats.seen += 1

    if not label.ok:
        stats.reasons[f"teacher-error: {label.error}"] += 1
        return None

    missing = grounded(label.values, span)
    if missing:
        stats.reasons[f"ungrounded: {','.join(sorted(missing)[:3])}"] += 1
        return None
    stats.grounded += 1

    problems = validated(label.values)
    if problems:
        stats.reasons[f"validator: {problems[0][:60]}"] += 1
        return None
    stats.validated += 1

    builder = MessageBuilder(model)
    slots = to_slots(label.values, model)
    # Fields the document cannot contain (MsgId, CreDtTm, NbOfTxs, CtrlSum) are
    # supplied here. Without this the build fails on a required MsgId, which looks
    # like a model failure and is actually a missing pipeline stage.
    fill_system_fields(slots, model)
    result = builder.build(slots)
    if not result.ok or not result.xml:
        reason = result.errors[0] if result.errors else "empty"
        stats.reasons[f"build: {reason[:60]}"] += 1
        return None

    xsd_ok, xsd_errors = validate(result.xml, model.path)
    if not xsd_ok:
        stats.reasons[f"xsd: {xsd_errors[0][:60] if xsd_errors else 'unknown'}"] += 1
        return None
    stats.serializable += 1
    stats.kept += 1

    return DistilledExample(span, label.values, label.reasoning, agreement, result.xml)


# ---------------------------------------------------------------- data emission


class MockTeacher:
    """Offline stand-in used by --selftest.

    Resolves schema keys by trailing element name, the way a real teacher sees
    them (key plus the schema path in the description). Each case targets one
    specific gate so the selftest proves the filters fire rather than merely that
    they run.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    @staticmethod
    def _key(schema: dict[str, Any], name: str) -> str | None:
        for key, entry in (schema.get("properties") or {}).items():
            if str(entry.get("description", "")).endswith("/" + name):
                return key
        return None

    def label(
        self,
        span: str,
        schema: dict[str, Any],
        shots: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> TeacherLabel:
        self.calls += 1
        key = lambda n: self._key(schema, n)  # noqa: E731
        nm, iban, amt, e2e = key("Nm"), key("IBAN"), key("InstdAmt"), key("EndToEndId")
        dt = key("Dt")
        ccy = key("Ccy")

        def build(pairs: dict[str | None, Any], reasoning: str) -> TeacherLabel:
            return TeacherLabel(
                {k: v for k, v in pairs.items() if k is not None}, reasoning
            )

        if "GB33" in span:  # IBAN checksum deliberately wrong -> validator gate
            return build(
                {nm: "Northwind Ltd", iban: "GB33BUKB20201555555556"},
                "'Northwind Ltd' -> Nm; IBAN copied verbatim",
            )
        if "JPY" in span:  # 2 dp on a 0-dp currency -> minor-unit gate
            return build(
                {nm: "ACME GmbH", amt: "125000.50", ccy: "JPY"},
                "'125000.50' -> amount",
            )
        if "Northwind" in span:  # clean, fully grounded -> survives
            return build(
                {
                    nm: "ACME GmbH",
                    iban: "DE89370400440532013000",
                    amt: "1250.00",
                    e2e: "E2E-1",
                    dt: "2026-09-20",
                    ccy: "EUR",
                },
                "'ACME GmbH' -> Nm; 'DE89...' -> IBAN; '1250.00' -> amount; "
                "'E2E-1' -> EndToEndId; '2026-09-20' -> execution date",
            )
        return build({nm: "Invented Corp"}, "fabricated")  # grounding gate


def to_needle_jsonl(example: DistilledExample, schema: dict[str, Any]) -> dict[str, Any]:
    """Emit a line in Needle's finetune JSONL format so it is directly trainable."""
    return {
        "query": example.span,
        "tools": [{"name": "extract_payment", "parameters": schema}],
        "answers": [{"name": "extract_payment", "arguments": example.values}],
        "reasoning": example.reasoning,
    }


# ------------------------------------------------------------------ orchestration


def selftest() -> int:
    schema_path = SCHEMAS / "pain.001.001.09.xsd"
    if not schema_path.exists():
        print(f"schema not found: {schema_path}")
        return 2

    model = XSDModel(schema_path)
    specs = field_specs(model)
    schema = json_schema_for(specs)
    print(f"derived {len(specs)} fields from the schema")
    print(f"sample  : {', '.join(s.name for s in specs[:8])} ...")

    teacher = MockTeacher()
    stats = FilterStats()
    kept: list[DistilledExample] = []

    spans = [
        "Northwind Ltd instructs payment of EUR 1250.00 to ACME GmbH, "
        "IBAN DE89370400440532013000, ref E2E-1, to settle on 2026-09-20.",
        "Pay GBP 10.00 to Northwind Ltd, IBAN GB33BUKB20201555555556.",
        "Settle JPY 125000.50 to ACME GmbH today.",
        "Some unrelated text about the weather in Lisbon.",
    ]
    for span in spans:
        label_obj = teacher.label(span, schema)
        out = distill_one(span, specs, model, label_obj, stats=stats)
        if out:
            kept.append(out)

    print()
    print("=" * 74)
    print(
        "FILTER FUNNEL (mock teacher: 1 clean, 1 bad checksum, 1 bad amount, 1 fabricated)"
    )
    print("=" * 74)
    print(f"  {stats.row()}")
    print()
    print("drop reasons:")
    for reason, count in stats.reasons.most_common():
        print(f"  {count}x  {reason}")

    print()
    print("=" * 74)
    print("KEPT EXAMPLES")
    print("=" * 74)
    for example in kept:
        print(f"  span  : {example.span[:64]}")
        print(f"  values: {json.dumps(example.values, ensure_ascii=False)}")
        print("  XSD   : valid")
        print()

    out_path = PACKAGE_ROOT / "distilled_selftest.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for example in kept:
            handle.write(json.dumps(to_needle_jsonl(example, schema)) + "\n")
    print(f"wrote {len(kept)} training lines -> {out_path}")

    ok = stats.kept == 1 and stats.seen == 4
    print()
    print(f"SELFTEST: {'PASS' if ok else 'FAIL'} (expected exactly 1 of 4 to survive)")
    return 0 if ok else 1


def run(
    span_file: str,
    out_file: str,
    schema_name: str = "pain.001.001.09.xsd",
    samples: int = 5,
) -> int:
    teacher = TeacherClient()
    model = XSDModel(SCHEMAS / schema_name)
    specs = field_specs(model)
    schema = json_schema_for(specs)
    stats = FilterStats()

    spans = [
        line.strip()
        for line in Path(span_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(
        f"{len(spans)} spans, {samples} teacher samples each ({len(spans) * samples} calls)"
    )

    kept = 0
    with Path(out_file).open("w", encoding="utf-8") as handle:
        for i, span in enumerate(spans, 1):
            agreement = teacher.label_consensus(span, schema, n=samples)
            label_obj = TeacherLabel(agreement.consensus, "")
            example = distill_one(
                span, specs, model, label_obj, agreement=agreement.agreement, stats=stats
            )
            if example:
                handle.write(json.dumps(to_needle_jsonl(example, schema)) + "\n")
                kept += 1
            if i % 100 == 0:
                print(f"  {i}/{len(spans)}  kept={kept}  {cost_report(teacher)}")

    print()
    print(stats.row())
    for reason, count in stats.reasons.most_common(10):
        print(f"  {count}x  {reason}")
    print(cost_report(teacher))
    print(f"wrote {kept} lines -> {out_file}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run the filters against a mock teacher (no API key)",
    )
    parser.add_argument("--spans", help="file of spans, one per line")
    parser.add_argument("--out", default="distilled.jsonl")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if not args.spans:
        parser.error("--spans is required unless --selftest")
    return run(args.spans, args.out, samples=args.samples)


if __name__ == "__main__":
    raise SystemExit(main())
