"""The canonical payload: field names, and parsing a generation into one.

This module exists to sit at the bottom of the dependency graph. It imports
nothing from `iso20022_lab`, and that is a requirement rather than a preference.

WHY IT WAS SPLIT OUT OF `acceptance.py`

The gate (`acceptance`), the training metric (`model.train`) and the evaluator
(`model.evaluate_model`) all need to turn a generated string into a payload and
check its key names. Those primitives lived in `acceptance.py`, so when the
training metric imported them it created:

    acceptance -> workflow -> model.evaluate_model -> model.train -> acceptance

pyright reported it as `Cycle detected in import chain` with exactly that path.
A deferred import inside a function would have silenced the report while leaving
the layering wrong, because the real problem is not the import statement: it is
that **pure functions over strings were living in the module that also knows
about trained models, XSD round-trips and human workflow.** Parsing `{...}` has
no reason to depend on any of that.

The extraction also settles a drift risk that had already bitten once. Three
parsers grew independently -- `acceptance.parse_strict`, `train._parse_json` and
`evaluate_model.parse_generation` -- with nothing keeping them in agreement, and
a duplicated `CANONICAL_KEYS` in a test file that could silently diverge from the
library's. There is now one copy of each, in a module nothing else can influence.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, TypeVar

E = TypeVar("E", bound=Mapping[str, Any])

# The nine field names the model is allowed to emit. Anything else is invented.
CANONICAL_KEYS: frozenset[str] = frozenset(
    {
        "Cdtr_Nm",
        "CdtrAcct_IBAN",
        "CdtrAgt_BICFI",
        "Dbtr_Nm",
        "DbtrAcct_IBAN",
        "Amt_InstdAmt",
        "Amt_InstdAmt_Ccy",
        "PmtId_EndToEndId",
        "ReqdExctnDt_Dt",
        "RmtInf_Ustrd",
    }
)


def first_object_body(raw: str) -> str:
    """Return the body of the first balanced `{...}` in `raw`, or "".

    Scans braces rather than slicing from the first `{` to the last `}`. That
    distinction is not cosmetic: this function used to do `find("{")` to
    `rfind("}")`, so when a generation emitted a valid object and then kept going
    -- which it did, because decoding was not stopping on the answer terminator --
    the trailing noise was swept *into* the body and the object failed to parse.
    The measurement then reported a model failure that was really a decode bug.

    Brace counting is depth-aware, so a `}` inside a string value does not end the
    scan early. Escapes inside strings are skipped for the same reason.
    """
    start = raw.find("{")
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return ""


def parse_strict(raw: str) -> tuple[dict[str, str], str]:
    """Parse a generation, returning (payload, how-it-failed).

    Stricter than the production parser on purpose. Production hunts for the
    first `{` and last `}` and returns `{}` on failure, which is right at
    inference time: a bad generation becomes a document for review rather than a
    crash. For acceptance we need to know *that* it failed and why, so `{}` is a
    failure here rather than a quiet no-op.
    """
    body = first_object_body(raw)
    if not body:
        return {}, "no JSON object in output"
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON: {exc.msg} at pos {exc.pos}"
    if not isinstance(parsed, dict):
        return {}, f"top-level {type(parsed).__name__}, expected object"
    return {str(k): str(v) for k, v in parsed.items()}, ""


def count_duplicate_keys(raw: str) -> int:
    """Count top-level keys appearing more than once in the raw text.

    `json.loads` keeps only the last of a repeated key, so duplicates vanish the
    moment you parse -- which is why this reads the text. The step-400 sample
    emitted `"Cdtr_Nm"` twice and `"PmtId_Nm"` twice; a dict-based check would
    have seen neither.

    Flat scan of the object body, which is correct for a nine-key flat schema and
    would need revisiting if nested objects were ever added.
    """
    body = first_object_body(raw)
    if not body:
        return 0
    keys = re.findall(r'"([^"\\]+)"\s*:', body[1:-1])
    return len(keys) - len(set(keys))


def canonical_only(payload: dict[str, str]) -> bool:
    """True when every key is a field the schema defines."""
    return set(payload) <= CANONICAL_KEYS


def stratified_sample(examples: list[E], limit: int) -> list[E]:
    """Take ``limit`` examples spread across every difficulty level.

    Corpus builders emit all the clean documents first, then all the messy ones,
    then all the hostile ones, so a plain ``examples[:n]`` slice is not a sample
    of the corpus -- it is the easy end of it.

    That shape of bug showed up twice and cost real accuracy both times. The
    training metric took ``examples[:8]``, which is seven clean documents, and so
    reported ``recall 100.0%`` for a model the acceptance gate scores at 71.7%
    over a difficulty-balanced sample. The gate took ``examples[:30]``, which is
    thirty clean documents, so it never scored ``messy`` or ``hostile`` at all --
    the two levels the model card calls the hard case. Neither number was a lie;
    both were answers to a narrower question than they appeared to answer.

    This lives in ``fields`` because that module imports nothing from the
    package. ``acceptance`` cannot import from ``model.train`` directly: the chain
    ``acceptance -> workflow -> model.evaluate_model -> model.train -> acceptance``
    is a genuine cycle, so a shared helper has to sit below both of them.

    Buckets are walked in sorted order so selection is deterministic, and the
    remainder is topped up from the front so a corpus smaller than
    ``limit * levels`` still returns what it has.
    """
    buckets: dict[str, list[E]] = {}
    for item in examples:
        buckets.setdefault(str(item.get("difficulty", "?")), []).append(item)
    if not buckets:
        return []

    per_bucket = max(1, limit // len(buckets))
    chosen: list[E] = []
    taken: set[int] = set()
    for name in sorted(buckets):
        for item in buckets[name][:per_bucket]:
            chosen.append(item)
            taken.add(id(item))
    for item in examples:
        if len(chosen) >= limit:
            break
        if id(item) not in taken:
            chosen.append(item)
            taken.add(id(item))
    return chosen[:limit]
