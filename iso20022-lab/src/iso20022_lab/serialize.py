"""
Schema-driven serialization: flat slots in, XSD-valid ISO 20022 XML out.

The whole point of this module is that the model never writes XML. The model
supplies *values* in a nested dict; this module supplies *structure* by walking
the element tree read from the XSD. Element order therefore comes from
xs:sequence, not from the model, so ordering bugs are impossible by
construction rather than merely unlikely.

Slots are a nested dict mirroring the message. Key order is irrelevant: the
serializer emits children in schema order, and any key the schema does not
declare is reported as unknown instead of being silently written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from lxml import etree  # type: ignore[attr-defined]  # lxml 6.x ships no etree stubs

from iso20022_lab.xsd_introspect import Element, XSDModel, walk_element

Value = str | int | float
# Mapping and Sequence, not dict and list: both are covariant in their element
# type, so a plain nested literal is assignable to this alias. dict and list are
# invariant, which would reject perfectly good slot trees. Covariance is also the
# honest description of how the serializer uses slots: it only ever reads.
Slots = Mapping[str, "Value | Slots | Sequence[Value | Slots]"]

# Fields a source document cannot contain, because they describe the message
# itself rather than the payment. A message is invalid without them, and a model
# asked to extract them will hallucinate a plausible value that then fails the
# grounding check. They are therefore SUPPLIED by the system at build time.
#
# This is the third field class. Getting it wrong is subtle: excluding these from
# extraction is only half the job, and the other half is filling them.
SYSTEM_FIELDS = frozenset(
    {"MsgId", "PmtInfId", "CreDtTm", "NbOfTxs", "CtrlSum", "BtchBookg", "Grpg"}
)


# Fields fixed by the MESSAGE TYPE rather than read from a document. A customer
# credit transfer IS a transfer, so asking a model to infer PmtMtd from prose
# invites a wrong answer to a question that has one right answer.
#
# This completes the field taxonomy, which only became visible by running the
# pipeline: EXTRACTED (from the document), GENERATED (assigned by the system),
# DERIVED (computed from other values), and DEFAULTED (fixed by the message type).
DEFAULT_FIELDS: dict[str, str] = {
    "PmtMtd": "TRF",
    "SttlmMtd": "INDA",
    "BtchBookg": "false",
}


def _system_value(
    name: str,
    slots: Slots,
    msg_id: str | None,
    now: str | None,
) -> str | None:
    """Produce a value for a system field, or None when it cannot be derived.

    Called once per target node so that identifier fields stay unique across
    repeated containers.
    """
    if name == "MsgId":
        return msg_id
    if name == "PmtInfId":
        return f"PMT-{uuid4().hex[:12].upper()}"
    if name == "CreDtTm":
        return now
    if name == "NbOfTxs":
        return str(_count_transactions(slots))
    if name == "CtrlSum":
        return _sum_amounts(slots)
    if name in DEFAULT_FIELDS:
        return DEFAULT_FIELDS[name]
    return None


def _iter_tx_inf(node: object) -> list[Mapping[str, object]]:
    out: list[Mapping[str, object]] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key == "CdtTrfTxInf":
                items = (
                    value
                    if isinstance(value, Sequence) and not isinstance(value, str)
                    else [value]
                )
                out.extend(i for i in items if isinstance(i, Mapping))
            else:
                out.extend(_iter_tx_inf(value))
    elif isinstance(node, Sequence) and not isinstance(node, str):
        for item in node:
            out.extend(_iter_tx_inf(item))
    return out


def _count_transactions(slots: Slots) -> int:
    return len(_iter_tx_inf(slots)) or 1


def _sum_amounts(slots: Slots) -> str:
    total = 0.0
    for tx in _iter_tx_inf(slots):
        amt = tx.get("Amt")
        if isinstance(amt, Mapping):
            inner = amt.get("InstdAmt")
            if isinstance(inner, Mapping):
                raw = inner.get("_") or inner.get("value")
                # Narrow explicitly: Mapping.get returns object, and float() will
                # not accept an arbitrary object.
                if isinstance(raw, (str, int, float)):
                    total += float(raw)
    return f"{total:.2f}"


def _resolve_targets(root: dict[str, Any], parts: list[str]) -> list[dict[str, Any]]:
    """Resolve every mutable mapping a path prefix points at.

    Repeating containers are lists, so a naive descent replaces the list with a
    fresh dict and destroys the transactions inside it. Fanning out over list
    elements is what makes this safe for `PmtInf` and `CdtTrfTxInf`, which are
    both 1..* and both carry required system fields.
    """
    nodes: list[dict[str, Any]] = [root]
    for part in parts:
        nxt: list[dict[str, Any]] = []
        for node in nodes:
            value = node.get(part)
            if isinstance(value, dict):
                nxt.append(value)
            elif isinstance(value, Sequence) and not isinstance(value, str):
                nxt.extend(item for item in value if isinstance(item, dict))
            else:
                created: dict[str, Any] = {}
                node[part] = created
                nxt.append(created)
        nodes = nxt
    return nodes


def fill_system_fields(
    slots: dict[str, Any],
    model: XSDModel,
    msg_id: str | None = None,
    now: str | None = None,
) -> list[str]:
    """Fill every required SYSTEM_FIELD path that the caller did not supply.

    Returns the paths filled. Without this step a message built from extracted
    values alone fails schema validation on MsgId, which looks like a model
    failure and is actually an architecture gap.
    """
    from datetime import datetime, timezone

    if msg_id is None:
        msg_id = f"MSG-{uuid4().hex[:16].upper()}"
    if now is None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    filled: list[str] = []
    for path, el in model.paths():
        if not el.required_here or not el.is_leaf:
            continue
        if el.name not in SYSTEM_FIELDS and el.name not in DEFAULT_FIELDS:
            continue
        parts = path.split("/")[1:]
        if not parts:
            continue
        for node in _resolve_targets(slots, parts[:-1]):
            if parts[-1] in node:
                continue
            value = _system_value(el.name, slots, msg_id, now)
            if value is None:
                continue
            node[parts[-1]] = value
            filled.append(path)
    return filled


@dataclass
class BuildResult:
    xml: str | None
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        state = "OK" if self.ok else "FAILED"
        lines = [f"build: {state}"]
        lines += [f"  error: {e}" for e in self.errors]
        lines += [f"  warn : {w}" for w in self.warnings]
        return "\n".join(lines)


class MessageBuilder:
    """Turns slots into XML whose shape is dictated by the schema."""

    def __init__(self, model: XSDModel):
        self.model = model
        self.ns = model.target_ns

    # ------------------------------------------------------------------ build

    def build(self, slots: Slots) -> BuildResult:
        errors: list[str] = []
        warnings: list[str] = []
        root_spec = self.model.message()

        document = etree.Element(f"{{{self.ns}}}Document", nsmap={None: self.ns})
        self._fill(document, root_spec, slots, "", errors, warnings)

        body = etree.tostring(
            document, pretty_print=True, xml_declaration=True, encoding="UTF-8"
        ).decode()
        return BuildResult(xml=body, ok=not errors, errors=errors, warnings=warnings)

    def _fill(
        self,
        parent_xml: etree._Element,
        spec: Element,
        slots: Slots,
        path: str,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        if spec.is_choice:
            self._fill_choice(parent_xml, spec, slots, path, errors, warnings)
        else:
            self._fill_sequence(parent_xml, spec, slots, path, errors, warnings)

        known = {c.name for c in spec.children}
        for key in slots:
            if key not in known:
                warnings.append(f"{path}/{key}: not declared by the schema (dropped)")

    def _fill_choice(
        self,
        parent_xml: etree._Element,
        spec: Element,
        slots: Slots,
        path: str,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Emit exactly one branch of an xs:choice.

        Emitting more than one branch is precisely what makes the XSD reject a
        document with "this element is not expected", so extra branches are
        reported and dropped rather than written into an invalid message.
        """
        supplied = [c for c in spec.children if c.name in slots]
        options = ", ".join(c.name for c in spec.children)

        if not supplied:
            if spec.required:
                errors.append(f"{path}: choose exactly one of [{options}]")
            return

        if len(supplied) > 1:
            chosen_names = ", ".join(c.name for c in supplied)
            errors.append(
                f"{path} is a choice (maxOccurs=1) but {len(supplied)} branches were "
                f"supplied [{chosen_names}]; emitting only '{supplied[0].name}'"
            )

        chosen = supplied[0]
        child_path = f"{path}/{chosen.name}" if path else chosen.name
        raw = slots[chosen.name]
        if chosen.is_leaf:
            self._emit_leaf(parent_xml, chosen, raw, child_path, errors)
            return
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            if not isinstance(item, Mapping):
                errors.append(f"{child_path} expects an object, got {type(item).__name__}")
                continue
            self._fill_child(parent_xml, chosen, item, child_path, errors, warnings)

    def _fill_sequence(
        self,
        parent_xml: etree._Element,
        spec: Element,
        slots: Slots,
        path: str,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        for child_spec in spec.children:
            child_path = f"{path}/{child_spec.name}" if path else child_spec.name
            raw = slots.get(child_spec.name)

            if raw is None:
                # required_here, not required: a branch of an enclosing xs:choice
                # is not individually mandatory.
                if child_spec.required_here and child_spec.is_leaf:
                    errors.append(f"{child_path} is required (minOccurs=1)")
                elif child_spec.children and child_spec.required_here:
                    self._fill_child(
                        parent_xml, child_spec, {}, child_path, errors, warnings
                    )
                continue

            if child_spec.is_leaf:
                self._emit_leaf(parent_xml, child_spec, raw, child_path, errors)
            else:
                items = raw if isinstance(raw, list) else [raw]
                if len(items) > 1 and not child_spec.repeatable:
                    errors.append(
                        f"{child_path} has {len(items)} values but maxOccurs="
                        f"{child_spec.max_occurs}"
                    )
                for item in items:
                    if not isinstance(item, Mapping):
                        errors.append(
                            f"{child_path} expects an object, got {type(item).__name__}"
                        )
                        continue
                    self._fill_child(
                        parent_xml, child_spec, item, child_path, errors, warnings
                    )

    def _fill_child(
        self,
        parent_xml: etree._Element,
        spec: Element,
        item: Slots,
        path: str,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        node = etree.SubElement(parent_xml, f"{{{self.ns}}}{spec.name}")
        self._fill(node, spec, item, path, errors, warnings)

    def _emit_leaf(
        self,
        parent_xml: etree._Element,
        spec: Element,
        raw: object,
        path: str,
        errors: list[str],
    ) -> None:
        values = raw if isinstance(raw, list) else [raw]
        if len(values) > 1 and not spec.repeatable:
            errors.append(f"{path}: multiple values but maxOccurs={spec.max_occurs}")
        for value in values:
            if isinstance(value, Mapping):
                amount = value.get("_value", value.get("value"))
                currency = value.get("Ccy") or value.get("ccy")
                node = etree.SubElement(parent_xml, f"{{{self.ns}}}{spec.name}")
                if currency:
                    node.set("Ccy", str(currency))
                node.text = "" if amount is None else str(amount)
                if (
                    spec.enumerations
                    and amount is not None
                    and str(amount) not in spec.enumerations
                ):
                    errors.append(f"{path}: '{amount}' not in {spec.type_name}")
                continue

            node = etree.SubElement(parent_xml, f"{{{self.ns}}}{spec.name}")
            node.text = str(value)
            if spec.enumerations and str(value) not in spec.enumerations:
                errors.append(
                    f"{path}: '{value}' not in code list {spec.type_name} "
                    f"(e.g. {', '.join(spec.enumerations[:5])})"
                )


def validate(xml: str | Path, xsd_path: str | Path) -> tuple[bool, list[str]]:
    """Validate a document against a real ISO 20022 XSD using libxml2."""
    schema = etree.XMLSchema(etree.parse(str(xsd_path)))
    doc = etree.fromstring(xml.encode()) if isinstance(xml, str) else etree.parse(str(xml))
    if schema.validate(doc):
        return True, []
    return False, [
        f"line {e.line}: {e.message.strip()}"
        for e in schema.error_log  # type: ignore[attr-defined]
    ]


def describe(model: XSDModel) -> str:
    """Human-readable summary of what the schema requires."""
    msg = model.message()
    paths = walk_element(msg)
    leaves = [p for p, el in paths if el.is_leaf]
    required = model.required_leaf_paths()
    codes = model.code_sets_for()
    out = [
        f"message     : {model.version}",
        f"namespace   : {model.target_ns}",
        f"elements    : {len(paths)}  ({len(leaves)} leaf)",
        f"required    : {len(required)} required leaf paths",
        f"code lists  : {len(codes)} internal",
    ]
    for name, values in sorted(codes.items())[:6]:
        out.append(f"   {name:<40} {len(values):>3} values")
    repeats = [(p, el) for p, el in paths if el.repeatable]
    if repeats:
        out.append(f"repeatable  : {len(repeats)} (e.g. {repeats[0][0]})")
    return "\n".join(out)
