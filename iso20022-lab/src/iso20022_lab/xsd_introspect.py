"""
ISO 20022 XSD introspection.

Reads an official ISO 20022 XSD and extracts the two things an
unstructured->structured pipeline actually needs:

  1. The element tree: name, type, cardinality, and *sequence order*.
     Element order is mandatory in ISO 20022 (xs:sequence) and is the single
     biggest reason an LLM emits invalid messages when asked to write XML.
  2. The internal code lists: simpleType enumerations with their allowed values.

Note: ISO 20022 *external* code sets are NOT in the XSD by design, so they can
be updated without a schema release. They must be supplied separately; see
EXTERNAL_CODE_SETS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree  # type: ignore[attr-defined]  # lxml 6.x ships no etree stubs

XS = "http://www.w3.org/2001/XMLSchema"


def _q(tag: str) -> str:
    """Qualify a local name into the XML Schema namespace."""
    return f"{{{XS}}}{tag}"


def _enum_values(parent: etree._Element) -> tuple[str, ...]:
    """All xs:enumeration values under `parent`, as a typed tuple of str."""
    values: list[str] = []
    for node in parent.findall(_q("enumeration")):
        value = node.get("value")
        if value is not None:
            values.append(value)
    return tuple(values)


def _children_by_tag(parent: etree._Element, tag: str) -> list[etree._Element]:
    return list(parent.findall(_q(tag)))


@dataclass
class Element:
    """One element in the message tree, in xs:sequence order."""

    name: str
    type_name: str
    min_occurs: int = 1
    max_occurs: int | None = 1  # None == unbounded
    children: list[Element] = field(default_factory=list)
    # True when this element's children are xs:choice alternatives rather than an
    # xs:sequence. At most ONE branch may be emitted; "required" then means "at
    # least one branch", not "every branch".
    is_choice: bool = False
    # True when this element is itself one branch of a parent's xs:choice. Its own
    # minOccurs governs repetition *within* the taken branch, not whether the
    # branch must be present, so required-field reporting must skip it.
    in_choice: bool = False
    enumerations: tuple[str, ...] = ()
    pattern: str | None = None
    attribute_names: tuple[str, ...] = ()

    @property
    def required(self) -> bool:
        return self.min_occurs > 0

    @property
    def required_here(self) -> bool:
        """Required for this branch to be valid on its own.

        An element inside a choice is not individually required: the requirement
        belongs to the choice group, and any one of its branches satisfies it.
        """
        return self.min_occurs > 0 and not self.in_choice

    @property
    def repeatable(self) -> bool:
        return self.max_occurs is None or self.max_occurs > 1

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def cardinality(self) -> str:
        hi = "*" if self.max_occurs is None else str(self.max_occurs)
        return f"{self.min_occurs}..{hi}"


def walk_element(el: Element, prefix: str = "") -> list[tuple[str, Element]]:
    """Flatten an Element subtree to (slash/path, element) pairs, depth first.

    Module-level rather than a method so the recursion has no `Self` type,
    which keeps static analysis honest about the return type.
    """
    path = f"{prefix}/{el.name}" if prefix else el.name
    out: list[tuple[str, Element]] = [(path, el)]
    for child in el.children:
        out.extend(walk_element(child, path))
    return out


class XSDModel:
    """An introspectable view over one ISO 20022 message schema."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.tree = etree.parse(str(self.path))
        self.root = self.tree.getroot()
        self.target_ns: str = self.root.get("targetNamespace") or ""
        self.version: str = self.path.stem  # e.g. pain.001.001.09

        complex_types: dict[str, etree._Element] = {}
        for ct in _children_by_tag(self.root, "complexType"):
            name = ct.get("name")
            if name is not None:
                complex_types[name] = ct
        self._complex = complex_types

        simple_types: dict[str, etree._Element] = {}
        for st in _children_by_tag(self.root, "simpleType"):
            name = st.get("name")
            if name is not None:
                simple_types[name] = st
        self._simple = simple_types

        top_elements: dict[str, etree._Element] = {}
        for el in _children_by_tag(self.root, "element"):
            name = el.get("name")
            if name is not None:
                top_elements[name] = el
        self._elements = top_elements

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _occurs(el: etree._Element) -> tuple[int, int | None]:
        mn = int(el.get("minOccurs") or "1")
        raw = el.get("maxOccurs") or "1"
        mx: int | None = None if raw == "unbounded" else int(raw)
        return mn, mx

    def code_lists(self) -> dict[str, tuple[str, ...]]:
        """Internal code lists: simpleType name -> allowed values."""
        out: dict[str, tuple[str, ...]] = {}
        for name, st in self._simple.items():
            restr = st.find(_q("restriction"))
            if restr is None:
                continue
            values = _enum_values(restr)
            if values:
                out[name] = values
        return out

    def _simple_facets(self, type_name: str) -> tuple[tuple[str, ...], str | None]:
        st = self._simple.get(type_name)
        if st is None:
            return (), None
        restr = st.find(_q("restriction"))
        if restr is None:
            return (), None
        pattern_el = restr.find(_q("pattern"))
        pattern = pattern_el.get("value") if pattern_el is not None else None
        return _enum_values(restr), pattern

    def _attribute_names(self, ct: etree._Element) -> tuple[str, ...]:
        names: list[str] = []
        for sc in _children_by_tag(ct, "simpleContent"):
            for ext in _children_by_tag(sc, "extension"):
                for attr in _children_by_tag(ext, "attribute"):
                    nm = attr.get("name")
                    if nm is not None:
                        names.append(nm)
        for attr in _children_by_tag(ct, "attribute"):
            nm = attr.get("name")
            if nm is not None:
                names.append(nm)
        return tuple(names)

    # ------------------------------------------------------------------ build

    def _build_element(self, el: etree._Element, seen: frozenset[str]) -> Element:
        mn, mx = self._occurs(el)
        name = el.get("name") or (el.get("ref") or "").split(":")[-1]
        type_name = el.get("type") or ""

        if not type_name:
            inline_complex = el.find(_q("complexType"))
            if inline_complex is not None:
                children, is_choice = self._build_children(inline_complex, seen)
                return Element(
                    name=name,
                    type_name=f"{name}Type",
                    min_occurs=mn,
                    max_occurs=mx,
                    children=children,
                    is_choice=is_choice,
                )
            inline_simple = el.find(_q("simpleType"))
            if inline_simple is not None:
                restr = inline_simple.find(_q("restriction"))
                return Element(
                    name,
                    "",
                    mn,
                    mx,
                    enumerations=_enum_values(restr) if restr is not None else (),
                )
            return Element(name, "", mn, mx)

        enums, pattern = self._simple_facets(type_name)
        if enums or pattern or type_name in self._simple:
            return Element(name, type_name, mn, mx, enumerations=enums, pattern=pattern)

        ct = self._complex.get(type_name)
        if ct is None or type_name in seen:
            # Unknown or recursive reference: treat as a leaf.
            return Element(name, type_name, mn, mx)

        children, is_choice = self._build_children(ct, seen | {type_name})
        return Element(
            name,
            type_name,
            mn,
            mx,
            children=children,
            is_choice=is_choice,
            attribute_names=self._attribute_names(ct),
        )

    def _build_children(
        self, ct: etree._Element, seen: frozenset[str]
    ) -> tuple[list[Element], bool]:
        """Return (children, is_choice).

        The is_choice flag is what stops the serializer emitting both branches of
        an either/or, which is the difference between a valid message and one the
        XSD rejects with "this element is not expected".
        """
        container = ct.find(_q("sequence"))
        is_choice = False
        if container is None:
            container = ct.find(_q("choice"))
            is_choice = container is not None
        if container is None:
            return [], False  # simpleContent extension: a leaf holding attributes

        out: list[Element] = []
        for child in container:
            tag = etree.QName(child).localname
            if tag == "element":
                built = self._build_element(child, seen)
                built.in_choice = is_choice
                out.append(built)
            elif tag in ("sequence", "choice"):
                nested_is_choice = tag == "choice"
                for sub in _children_by_tag(child, "element"):
                    built = self._build_element(sub, seen)
                    built.in_choice = is_choice or nested_is_choice
                    out.append(built)
        return out, is_choice

    def message(self) -> Element:
        """The message root: Document -> <message>, e.g. CstmrCdtTrfInitn."""
        doc = self._elements.get("Document")
        if doc is None:
            raise ValueError(f"{self.path.name}: no top-level Document element")
        return self._build_element(doc, frozenset())

    # ------------------------------------------------------------- reporting

    def paths(self, message: Element | None = None) -> list[tuple[str, Element]]:
        return walk_element(message or self.message())

    def code_sets_for(self, message: Element | None = None) -> dict[str, tuple[str, ...]]:
        """Code lists reachable in this message, keyed by the type that uses them."""
        reachable: set[str] = set()
        for _path, el in self.paths(message):
            if el.enumerations:
                reachable.add(el.type_name or el.name)
        return {k: v for k, v in self.code_lists().items() if k in reachable}

    def required_leaf_paths(self, message: Element | None = None) -> list[str]:
        """Leaf paths that must be supplied for the message to validate.

        Choice branches are excluded: satisfying the *group* is what matters, and
        any one branch does that, so demanding all of them would be wrong.
        """
        return [p for p, el in self.paths(message) if el.required_here and el.is_leaf]

    def choice_groups(self, message: Element | None = None) -> list[str]:
        """Paths where exactly one of several branches must be chosen."""
        return [p for p, el in self.paths(message) if el.is_choice]

    def leaf_paths(self, message: Element | None = None) -> list[str]:
        return [p for p, el in self.paths(message) if el.is_leaf]


# ISO 20022 external code sets live outside the XSD by design. Subset needed for
# pain.001 / pacs.008; replace with the full published code set in production.
EXTERNAL_CODE_SETS: dict[str, tuple[str, ...]] = {
    "ExternalChargeBearerType1Code": ("DEBT", "CRED", "SHAR", "SLEV"),
    "ExternalPurpose1Code": (
        "SALA",
        "SUPP",
        "TAXS",
        "TRAD",
        "TREA",
        "VATX",
        "GDDS",
        "SCVE",
        "INTC",
        "RENT",
        "LOAN",
        "DIVI",
        "OTHR",
    ),
    "ExternalCategoryPurpose1Code": (
        "BONU",
        "CASH",
        "CORT",
        "DIVI",
        "GOVT",
        "HEDG",
        "INTC",
        "LOAN",
        "PENS",
        "SALA",
        "SECU",
        "SSBE",
        "SUPP",
        "TAXS",
        "TRAD",
        "TREA",
        "VATX",
        "WHLD",
    ),
    "ExternalServiceLevel1Code": ("SEPA", "G001", "G002", "NURG", "URGP", "SDVA"),
    "ExternalLocalInstrument1Code": ("CARD", "CHCK", "CSDC", "CSDB", "BACS", "BBDD"),
}


if __name__ == "__main__":
    import sys

    model = XSDModel(sys.argv[1])
    msg = model.message()
    print(f"schema     : {model.version}")
    print(f"namespace  : {model.target_ns}")
    print(f"elements   : {len(walk_element(msg))}")
    print(f"code lists : {len(model.code_lists())} internal")
    print(f"required   : {len(model.required_leaf_paths())} required leaf paths")
    print("\nfirst 30 paths:")
    for p, el in walk_element(msg)[:30]:
        mark = " " if el.required else "?"
        enum = f"  enum={len(el.enumerations)}" if el.enumerations else ""
        print(f"  {mark} {p:<56} [{el.cardinality:>4}]{enum}")
