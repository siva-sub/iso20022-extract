"""The rule-based baseline: what an institution already has today.

PROCESS.md 3.3 says the STP claim needs a measured exception rate. This supplies
the denominator: the extractor that is currently doing the work.

This is deliberately NOT a straw man. A real rule engine in a payment shop is
not one regex -- it is label-anchored extraction with a competent normaliser per
format, plus pattern fallbacks for unlabelled values, plus deterministic
validators to reject implausible candidates. All of that is implemented here,
because a weak baseline would flatter the model comparison and make the whole
measurement worthless.

What it does NOT have, and cannot have, is **role disambiguation**. Given

    Account: DE89370400440532013000

the rules know the value is an IBAN. They cannot know whether that account is
the payer's or the beneficiary's, because that fact lives in the document's
meaning and not in its surface form. When a label is missing, the pattern
fallback must guess by position, and it is wrong roughly half the time. That
single limitation is what the exception rate below is made of, and it is the
honest reason a rules engine plateaus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from iso20022_lab.corpus import FIELD_ALIASES
from iso20022_lab.validators import iban_valid, bic_valid

# --------------------------------------------------------------------------
# Value shapes. Ordered so the most specific pattern wins.
# --------------------------------------------------------------------------

IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
BIC_RE = re.compile(r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")
UETR_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
AMOUNT_RE = re.compile(r"\d[\d.,\u00a0 ]*\d|\d")
DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
DATE_DMY_RE = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b")
DATE_TEXT_RE = re.compile(
    r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})\b",
    re.IGNORECASE,
)

MONTHS: dict[str, str] = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}

# Labels that name a value but say nothing about its role. A document using one
# of these cannot be disambiguated by rules at all.
ROLE_AMBIGUOUS_LABELS: frozenset[str] = frozenset(
    {"account", "a/c", "acct no", "iban", "bic", "swift", "swift/bic", "reference", "ref"}
)


def _norm_amount(raw: str) -> str:
    """Normalise an amount to a period decimal, resolving separator convention.

    '1.250,00' and '1,250.00' are both 1250.00. The rule is: whichever separator
    appears LAST is the decimal separator, and the other one groups thousands.
    But '1250.00' has only one separator and it is a decimal one, while '1,250'
    has only one and it is a grouping one -- so a group of exactly three digits
    after a single separator is treated as grouping when no other separator is
    present AND the trailing group is exactly three digits.
    """
    text = raw.strip().replace("\u00a0", "").replace(" ", "")
    if not text:
        return ""
    has_dot = "." in text
    has_comma = "," in text
    if has_dot and has_comma:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif has_comma:
        head, _, tail = text.rpartition(",")
        if len(tail) == 3 and head:
            text = text.replace(",", "")
        else:
            text = text.replace(",", ".")
    elif has_dot:
        head, _, tail = text.rpartition(".")
        if len(tail) == 3 and head and head.count(".") >= 1:
            text = text.replace(".", "")
    try:
        return f"{float(text):.2f}"
    except ValueError:
        return ""


def _norm_date(raw: str) -> str:
    """Normalise a date to ISO 8601.

    Numeric ambiguity is resolved in favour of day-first, which is the
    convention in the documents this corpus models. That choice is recorded
    rather than hidden: a US-format document would be read wrong, and that is a
    real limitation of a rules engine, not a bug in this function.
    """
    text = raw.strip()
    if m := DATE_ISO_RE.fullmatch(text):
        return text
    if m := DATE_TEXT_RE.fullmatch(text):
        day, month_name, year = m.groups()
        return f"{year}-{MONTHS[month_name[:3].lower()]}-{int(day):02d}"
    if m := DATE_DMY_RE.fullmatch(text):
        a, b, year = m.groups()
        if int(a) > 12:
            day, month = int(a), int(b)
        elif int(b) > 12:
            month, day = int(a), int(b)
        else:
            day, month = int(a), int(b)
        return f"{year}-{month:02d}-{day:02d}"
    return ""


@dataclass
class Extraction:
    """Values found, plus the method used, so failures can be classified."""

    values: dict[str, str] = field(default_factory=dict)
    method: dict[str, str] = field(default_factory=dict)

    def get(self, key: str) -> str:
        return self.values.get(key, "")


class RuleExtractor:
    """Label-anchored extraction with pattern fallback.

    `aliases` maps canonical keys to the labels that may introduce them, taken
    from the corpus so both approaches face the same vocabulary.
    """

    def __init__(self, aliases: dict[str, tuple[str, ...]] | None = None) -> None:
        self.aliases = aliases or FIELD_ALIASES
        self._label_index: list[tuple[re.Pattern[str], str, bool]] = []
        for key, labels in self.aliases.items():
            for label in labels:
                escaped = re.escape(label)
                pattern = re.compile(
                    rf"^\s*{escaped}\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
                )
                self._label_index.append(
                    (pattern, key, label.lower() in ROLE_AMBIGUOUS_LABELS)
                )

    # ---------------------------------------------------------------- labels

    def _by_label(self, text: str) -> Extraction:
        found = Extraction()
        for pattern, key, ambiguous in self._label_index:
            if key in found.values:
                continue
            match = pattern.search(text)
            if not match:
                continue
            value = match.group(1).strip()
            value = self._coerce(key, value)
            if value:
                found.values[key] = value
                found.method[key] = "label-ambiguous" if ambiguous else "label"
        return found

    def _coerce(self, key: str, raw: str) -> str:
        """Normalise a labelled value to canonical form."""
        if key == "Amt_InstdAmt":
            # The value may carry the currency inline: "EUR 1,250.00".
            token = raw.split()[-1] if raw.split() else raw
            return _norm_amount(token)
        if key == "Amt_InstdAmt_Ccy":
            return self._currency_from(raw)
        if key == "ReqdExctnDt_Dt":
            return _norm_date(raw) or self._date_in(raw)
        if key == "CdtrAcct_IBAN" or key == "DbtrAcct_IBAN":
            m = IBAN_RE.search(raw.replace(" ", ""))
            return m.group(0) if m else ""
        if key.endswith("_BICFI"):
            m = BIC_RE.search(raw.replace(" ", ""))
            return m.group(0) if m and bic_valid(m.group(0)) else ""
        if key == "PmtId_EndToEndId":
            return raw.split()[0].strip(",;") if raw.split() else ""
        return raw

    @staticmethod
    def _currency_from(raw: str) -> str:
        for token in raw.replace(",", " ").split():
            if len(token) == 3 and token.isalpha() and token.isupper():
                return token
        return ""

    @staticmethod
    def _date_in(raw: str) -> str:
        for pattern in (DATE_ISO_RE, DATE_TEXT_RE, DATE_DMY_RE):
            m = pattern.search(raw)
            if m:
                return _norm_date(m.group(0))
        return ""

    # -------------------------------------------------------------- patterns

    def _by_pattern(self, text: str, found: Extraction) -> None:
        """Fill gaps from value shape alone.

        This is where role ambiguity bites. Two IBANs with no labels are ordered
        by appearance and assigned to beneficiary then payer; the assignment is
        a coin flip on any document that lists the payer first, and it cannot be
        resolved from the text at all.
        """
        lines = text.splitlines()

        if "Amt_InstdAmt_Ccy" not in found.values:
            for key in ("Amt_InstdAmt", "ReqdExctnDt_Dt"):
                for line in lines:
                    ccy = self._currency_from(line)
                    if ccy and len(ccy) == 3 and ccy.isalpha():
                        found.values["Amt_InstdAmt_Ccy"] = ccy
                        found.method["Amt_InstdAmt_Ccy"] = "pattern"
                        break
                if "Amt_InstdAmt_Ccy" in found.values:
                    break

        if "Amt_InstdAmt" not in found.values:
            candidates: list[str] = []
            for line in lines:
                for token in AMOUNT_RE.findall(line):
                    normalised = _norm_amount(token)
                    if not normalised:
                        continue
                    integer = normalised.split(".")[0].lstrip("0")
                    # A long digit run is an account, phone or invoice number,
                    # not a payment amount.
                    if len(integer) > 7:
                        continue
                    candidates.append(normalised)
            if candidates:
                # Prefer a value with a decimal part: money is written with one.
                decimals = [c for c in candidates if "." in c]
                chosen = decimals[0] if decimals else candidates[0]
                found.values["Amt_InstdAmt"] = chosen
                found.method["Amt_InstdAmt"] = "pattern"

        ibans: list[str] = []
        for token in IBAN_RE.findall(text.replace(" ", "")):
            if iban_valid(token) and token not in ibans:
                ibans.append(token)
        for key, value in zip(("CdtrAcct_IBAN", "DbtrAcct_IBAN"), ibans, strict=False):
            if key not in found.values:
                found.values[key] = value
                found.method[key] = "pattern-guess"

        bics = [b for b in BIC_RE.findall(text) if bic_valid(b)]
        if "CdtrAgt_BICFI" not in found.values and bics:
            found.values["CdtrAgt_BICFI"] = bics[0]
            found.method["CdtrAgt_BICFI"] = "pattern-guess"

        if "ReqdExctnDt_Dt" not in found.values:
            for line in lines:
                if d := self._date_in(line):
                    found.values["ReqdExctnDt_Dt"] = d
                    found.method["ReqdExctnDt_Dt"] = "pattern"
                    break

        if "PmtId_EndToEndId" not in found.values:
            for line in lines:
                for token in re.findall(r"\b[A-Z]{2,4}[-/]?\d{3,}\b", line):
                    found.values["PmtId_EndToEndId"] = token
                    found.method["PmtId_EndToEndId"] = "pattern"
                    break
                if "PmtId_EndToEndId" in found.values:
                    break

    def extract(self, text: str) -> Extraction:
        found = self._by_label(text)
        self._by_pattern(text, found)
        return found


def status() -> str:
    ex = RuleExtractor()
    return (
        f"rule baseline: {len(ex.aliases)} field keys, "
        f"{len(ex._label_index)} label patterns, "
        f"role-ambiguous labels: {len(ROLE_AMBIGUOUS_LABELS)}"
    )


def main() -> int:
    from iso20022_lab.corpus import DIFFICULTIES, render_document

    print(status())
    print()
    values = {
        "Cdtr_Nm": "ACME GmbH",
        "CdtrAcct_IBAN": "DE89370400440532013000",
        "CdtrAgt_BICFI": "DEUTDEFF",
        "Dbtr_Nm": "Northwind Trading Ltd",
        "DbtrAcct_IBAN": "GB33BUKB20201555555556",
        "Amt_InstdAmt": "1250.00",
        "Amt_InstdAmt_Ccy": "EUR",
        "PmtId_EndToEndId": "E2E-1",
        "ReqdExctnDt_Dt": "2026-09-20",
    }
    extractor = RuleExtractor()
    for difficulty in DIFFICULTIES:
        doc = render_document(values, "demo", difficulty=difficulty, seed=5)
        found = extractor.extract(doc.text)
        print(f"--- {difficulty} ---")
        print(doc.text)
        for key in doc.truth:
            got = found.get(key)
            mark = "ok " if got == doc.truth[key] else "MISS"
            print(
                f"  {mark} {key:<20} want={doc.truth[key]:<34} got={got!r} [{found.method.get(key, '-')}]"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
