"""
Teacher model client for distillation.

The teacher is DeepSeek Flash on the native DeepSeek API (https://api.deepseek.com),
which is OpenAI-compatible. Needle already defaults to a DeepSeek Flash model for
data generation, so this keeps the teacher consistent across the stack.

Credentials are read from the environment, never hard-coded. Put them in a
`.env` file at the repo root (gitignored) as:

    DEEPSEEK_API_KEY=...
    DEEPSEEK_BASE_URL=https://api.deepseek.com

One constraint shapes this entire module: an API teacher returns TEXT, not
logits. Classical Hinton distillation (match teacher logits at temperature T) is
therefore impossible here. What is available is **sequence-level distillation**
(Kim & Rush 2016, arXiv 1606.07947): the teacher emits output sequences, and the
student is trained to reproduce them.

Losing soft targets costs real signal, so this module recovers part of it by
sampling the teacher N times per input and treating agreement as a pseudo
distribution. Disagreement is not discarded silently -- it becomes a confidence
feature and a filter.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class TeacherAuthError(RuntimeError):
    """Missing or rejected credentials."""


def _load_dotenv() -> None:
    """Load a repo-root .env into os.environ without adding a dependency.

    Existing environment variables always win, so an exported key is never
    silently overridden by a stale file.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if candidate.is_file():
            for raw in candidate.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
            return


_load_dotenv()

# Native DeepSeek API. OpenAI-compatible, so only the base URL and model id differ
# from the OpenRouter path Needle uses by default.
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_TEACHER = "deepseek-flash"


def _endpoint_and_key(api_key: str | None, base_url: str | None) -> tuple[str, str]:
    """Resolve (endpoint, key), preferring the native DeepSeek API."""
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise TeacherAuthError(
            "no credentials found; set DEEPSEEK_API_KEY (or OPENROUTER_API_KEY), "
            "or create a gitignored .env at the repo root"
        )
    base = base_url or DEEPSEEK_BASE_URL
    endpoint = base.rstrip("/") + "/chat/completions"
    return endpoint, api_key


# The teacher's job is deliberately narrow: read one span, emit values that are
# strictly evidenced in it, plus a short derivation per value naming the source
# text. The derivation is the point -- it is the grounding supervision the
# student otherwise never receives, and grounding is where the small model fails
# (see PROCESS.md section 2.5).
_TEACHER_SYSTEM = (
    "You extract structured payment fields from a single span of text. You may "
    "only emit values that appear in the span. If a field has no evidence in the "
    "span, omit it. Never guess, never fill placeholders, never use outside "
    "knowledge. Return only JSON."
)

_TEACHER_TEMPLATE = """Field schema (JSON Schema):
{schema}

Reference examples of correct extractions:
{shots}

Span:
---
{span}
---

Return a JSON object with exactly two keys:
  "reasoning": one short line per extracted field, of the form
               "'<source text>' -> <field>", quoting the span text the value came from
  "values":    an object mapping field name to the extracted value, containing
               ONLY fields with evidence in the span

If no field has evidence, return {{"reasoning": "", "values": {{}}}}.
Return only the JSON object."""


@dataclass
class TeacherLabel:
    """One teacher attempt at labelling a span."""

    values: dict[str, Any]
    reasoning: str
    raw: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class Agreement:
    """Agreement across repeated teacher samples for one span.

    Stands in for the soft label an API teacher cannot provide.
    """

    consensus: dict[str, Any]
    agreement: float  # 0..1, mean per-field modal agreement
    samples: list[TeacherLabel] = field(default_factory=list)
    disagreements: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def unanimous(self) -> bool:
        return self.agreement >= 1.0


class TeacherClient:
    """DeepSeek Flash, native API by default (OpenAI-compatible)."""

    def __init__(
        self,
        model: str = DEFAULT_TEACHER,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 180,
    ):
        self.endpoint, self.api_key = _endpoint_and_key(api_key, base_url)
        self.model = model
        self.timeout = timeout
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def _post(self, messages: list[dict[str, str]], temperature: float) -> str:
        payload = json.dumps(
            {"model": self.model, "messages": messages, "temperature": temperature}
        ).encode()
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode())
        usage = body.get("usage") or {}
        self.input_tokens += int(usage.get("prompt_tokens") or 0)
        self.output_tokens += int(usage.get("completion_tokens") or 0)
        self.calls += 1
        return body["choices"][0]["message"]["content"]

    def label(
        self,
        span: str,
        schema: dict[str, Any],
        shots: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> TeacherLabel:
        """Ask the teacher to label one span."""
        prompt = _TEACHER_TEMPLATE.format(
            schema=json.dumps(schema, indent=2),
            shots=json.dumps(shots or [], indent=2),
            span=span,
        )
        try:
            raw = self._post(
                [
                    {"role": "system", "content": _TEACHER_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature,
            )
        except (urllib.error.URLError, TimeoutError, KeyError) as exc:
            return TeacherLabel({}, "", error=f"{type(exc).__name__}: {exc}")

        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return TeacherLabel({}, "", raw=raw, error="no JSON object in reply")
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            return TeacherLabel({}, "", raw=raw, error=f"bad JSON: {exc}")

        values = parsed.get("values")
        if not isinstance(values, dict):
            return TeacherLabel({}, "", raw=raw, error="missing 'values' object")
        return TeacherLabel(values, str(parsed.get("reasoning") or ""), raw=raw)

    def label_consensus(
        self,
        span: str,
        schema: dict[str, Any],
        shots: list[dict[str, Any]] | None = None,
        n: int = 5,
        temperature: float = 0.7,
    ) -> Agreement:
        """Sample the teacher n times and reduce to a consensus.

        Self-consistency is the replacement for missing soft labels. Unanimity is
        a strong quality signal; disagreement is informative rather than merely
        noisy, so it is recorded per field instead of being averaged away.
        """
        samples = [
            self.label(span, schema, shots, temperature=temperature) for _ in range(n)
        ]
        good = [s for s in samples if s.ok]
        if not good:
            return Agreement({}, 0.0, samples)

        fields: set[str] = set()
        for label in good:
            fields |= set(label.values)

        consensus: dict[str, Any] = {}
        disagreements: dict[str, list[Any]] = {}
        scores: list[float] = []
        for name in sorted(fields):
            seen = [
                json.dumps(label.values[name], sort_keys=True)
                for label in good
                if name in label.values
            ]
            if not seen:
                continue
            counts = Counter(seen)
            top, votes = counts.most_common(1)[0]
            consensus[name] = json.loads(top)
            scores.append(votes / len(good))
            if len(counts) > 1:
                disagreements[name] = [json.loads(k) for k in counts]

        return Agreement(
            consensus=consensus,
            agreement=(sum(scores) / len(scores)) if scores else 0.0,
            samples=samples,
            disagreements=disagreements,
        )


def cost_report(client: TeacherClient) -> str:
    """Token usage so the one-time teacher cost stays visible."""
    return (
        f"teacher calls={client.calls}  input={client.input_tokens:,} tok  "
        f"output={client.output_tokens:,} tok"
    )
