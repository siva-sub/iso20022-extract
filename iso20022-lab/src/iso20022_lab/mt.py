"""
SWIFT MT <-> ISO 20022 MX translation.

This module exists because of the economic case in PROCESS.md section 1: the
CBPR+ migration is dated and mandated, MT and MX must dual-run during the
transition, and the failure mode is a message that validates and is semantically
wrong.

The mapping below is derived from a verified golden pair shipped in
`Query-farm/vgi-iso20022/data/dual/`, not from documentation. Both files describe
the same payment, so the translation rules are observed rather than assumed.

Five traps are visible in that single pair, and every one of them is the kind of
thing that survives a naive field-by-field mapper:

  1. Charge bearer vocabulary differs. MT uses SHA, MX uses SHAR. Same concept,
     different code, and an unrecognised code is a rejection.
  2. Amounts use a comma decimal separator in MT (`1234,56`) and a period in MX
     (`1234.56`). Naively parsing the MT string yields 123456 or an error.
  3. Dates are YYMMDD in MT (`260101`) and ISO 8601 in MX (`2026-01-01`).
  4. End-to-end reference is nested inside MT free text at `:72:/EToE/`. It is not
     its own tag, so a tag-dictionary mapper misses it entirely.
  5. MX carries `Purp/Cd` (purpose code) that has no MT103 source. It is inferred
     or enriched in transit. A model can learn that; a mapper can only default it.

Public MT corpora are tiny for the same reason paired extraction data is tiny:
real messages carry PII. The golden pair is a seed, not a dataset. The renderer
here is therefore the point -- it turns the observed mapping into unlimited
synthetic pairs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

# MT charge bearer -> MX ChrgBr. The vocabularies genuinely differ.
CHARGE_BEARER_MT_TO_MX = {
    "SHA": "SHAR",
    "OUR": "DEBT",
    "BEN": "CRED",
    "SLEV": "SLEV",
    "DEBT": "DEBT",
    "CRED": "CRED",
    "SHAR": "SHAR",
}
CHARGE_BEARER_MX_TO_MT = {v: k for k, v in CHARGE_BEARER_MT_TO_MX.items()}
CHARGE_BEARER_MX_TO_MT["SHAR"] = "SHA"  # canonical round-trip

MT_TAG_RE = re.compile(r"^:(\d{2}[A-Z]?):(.*)$")


def normalise_bic(bic: str, to_mx: bool = True) -> str:
    """Normalise a BIC between MT and MX conventions.

    MT headers conventionally carry the 11-character form with a branch code, and
    `XXX` means the primary office. MX BICFI in practice carries 8. Round-tripping
    without normalising makes otherwise identical fields compare unequal, which is
    exactly the false positive that breaks a dual-running check.
    """
    bic = bic.strip().upper()
    if to_mx:
        # XXX is the primary office, so the 8-character BIC is the same entity.
        if len(bic) == 11 and bic.endswith("XXX"):
            return bic[:8]
        return bic
    # to MT: pad a bare 8-character BIC out to the 11-character header form
    return bic + "XXX" if len(bic) == 8 else bic


@dataclass
class Mt103:
    """Field dictionary for a parsed MT103."""

    tags: dict[str, str] = field(default_factory=dict)
    sender_bic: str = ""
    receiver_bic: str = ""
    uetr: str = ""

    def get(self, tag: str, default: str = "") -> str:
        return self.tags.get(tag, default)

    @property
    def amount(self) -> tuple[str, str]:
        """(currency, amount) from :32A, normalised to a period decimal."""
        raw = self.get("32A")
        if len(raw) < 10:
            return "", ""
        currency = raw[6:9]
        amount = raw[9:].replace(",", ".")
        return currency, amount

    @property
    def settlement_date(self) -> str:
        """ISO 8601 date from the YYMMDD prefix of :32A."""
        raw = self.get("32A")
        if len(raw) < 6:
            return ""
        try:
            parsed = datetime.strptime(raw[:6], "%y%m%d")
        except ValueError:
            return ""
        return parsed.strftime("%Y-%m-%d")

    @property
    def end_to_end_id(self) -> str:
        """End-to-end reference, which MT hides inside :72 free text."""
        return _extract_e2e(self.get("72"))


def _extract_e2e(field72: str) -> str:
    """Pull the E2E reference out of :72, where MT nests it as /EToE/<ref>."""
    match = re.search(r"/EToE/([A-Za-z0-9\-]+)", field72)
    return match.group(1) if match else ""


def _split_party(block: str) -> tuple[str, str]:
    """Split an MT party field into (account, name).

    :50K and :59 are free text where an optional leading `/account` line is
    followed by name and address lines. The account is the part after the slash;
    the name is the next non-empty line.
    """
    lines = [ln.strip() for ln in block.splitlines()]
    lines = [ln for ln in lines if ln]
    account = ""
    if lines and lines[0].startswith("/"):
        account = lines[0].lstrip("/").strip()
        lines = lines[1:]
    name = lines[0] if lines else ""
    return account, name


def parse_mt103(text: str) -> Mt103:
    """Parse an MT103 into its tags plus the block-1/2/3 metadata.

    Block boundaries are located by prefix rather than by a single regex over the
    whole message, because block 3 contains NESTED braces (`{3:{121:<uuid>}}`) and
    block 4 is terminated by `-}`. A non-greedy `{n:...}` scan stops at the inner
    brace of block 3 and then fails its lookahead, which silently loses block 4 --
    every tag in the message.
    """
    parsed = Mt103()

    block1 = re.search(r"\{1:([^}]*)\}", text)
    if block1:
        match = re.search(r"F01([A-Z0-9]{8,11})", block1.group(1))
        if match:
            parsed.sender_bic = match.group(1)

    block2 = re.search(r"\{2:([^}]*)\}", text)
    if block2:
        match = re.search(r"I103([A-Z0-9]{8,11})", block2.group(1))
        if match:
            parsed.receiver_bic = match.group(1)

    block3 = re.search(r"\{3:\{(.*?)\}\}", text, re.DOTALL)
    if block3:
        # The capture starts AFTER `{3:{`, so the field is `121:<uuid>` with no
        # leading brace. Requiring one silently loses the UETR -- which is the
        # single most important field for gpi tracking and reconciliation.
        match = re.search(r"121:([0-9a-fA-F\-]+)", block3.group(1))
        if match:
            parsed.uetr = match.group(1).strip()

    if "{4:" in text:
        body = text.split("{4:", 1)[1]
        # Block 4 ends with a line containing only `-}`.
        for terminator in ("\n-}", "-}"):
            if terminator in body:
                body = body.rsplit(terminator, 1)[0]
                break
        parsed.tags = _parse_tags(body)
    return parsed


def _parse_tags(body: str) -> dict[str, str]:
    """Parse block-4 tags, attaching continuation lines to the open tag.

    MT party fields are multi-line (`:50K:/<iban>` then name then address), so a
    line-by-line scan must carry the current tag forward. Treating each line
    independently loses the name and address entirely.
    """
    tags: dict[str, str] = {}
    current = ""  # empty means "no open tag yet"; avoids a str | None key
    for line in body.splitlines():
        match = MT_TAG_RE.match(line)
        if match is not None:
            current = str(match.group(1))
            tags[current] = str(match.group(2) or "").strip()
        elif current and line.strip():
            tags[current] = f"{tags[current]}\n{line.strip()}"
    return tags


# ------------------------------------------------------------------ MT -> MX

# Observed MT103 tag -> ISO 20022 pacs.008 path. Every entry is evidenced by the
# golden pair rather than taken from documentation.
MT103_TO_MX = {
    "20": "PmtId/InstrId",  # transaction reference
    "70": "RmtInf/Ustrd",  # remittance information
    "32A": "IntrBkSttlmAmt",  # currency + amount (+ date, handled separately)
}


def mt103_to_dict(mt: Mt103) -> dict[str, str]:
    """Flatten an MT103 into the same key space the extraction pipeline uses.

    Keys deliberately mirror the MX field keys from `distill.field_key`, so an
    MT103 and a document become the same training input. That is what lets one
    student serve both the extraction and the migration tasks.
    """
    out: dict[str, str] = {}

    currency, amount = mt.amount
    if currency:
        out["Amt_IntrBkSttlmAmt_Ccy"] = currency
    if amount:
        out["Amt_IntrBkSttlmAmt"] = amount

    if mt.get("32A"):
        out["IntrBkSttlmDt"] = mt.settlement_date
    if mt.get("20"):
        out["PmtId_InstrId"] = mt.get("20")
    e2e = mt.end_to_end_id
    if e2e:
        out["PmtId_EndToEndId"] = e2e
    if mt.uetr:
        out["PmtId_UETR"] = mt.uetr

    dbtr_acct, dbtr_name = _split_party(mt.get("50K"))
    if dbtr_acct:
        out["DbtrAcct_IBAN"] = dbtr_acct
    if dbtr_name:
        out["Dbtr_Nm"] = dbtr_name

    cdtr_acct, cdtr_name = _split_party(mt.get("59"))
    if cdtr_acct:
        out["CdtrAcct_IBAN"] = cdtr_acct
    if cdtr_name:
        out["Cdtr_Nm"] = cdtr_name

    if mt.sender_bic:
        # Block 1/2 BICs are the MESSAGING endpoints (the banks on the SWIFT
        # network), not the account-servicing institutions. Mapping them onto
        # DbtrAgt/CdtrAgt is a classic error: in the golden pair the sender is
        # ACMEDEFF while the debtor's own agent is DEUTDEFF, so the naive mapping
        # produces two wrong fields that still look plausible.
        out["InstgAgt_BICFI"] = normalise_bic(mt.sender_bic, to_mx=True)
    if mt.receiver_bic:
        out["InstdAgt_BICFI"] = normalise_bic(mt.receiver_bic, to_mx=True)

    charge = mt.get("71A").strip().upper()
    if charge:
        out["ChrgBr"] = CHARGE_BEARER_MT_TO_MX.get(charge, charge)
    if mt.get("70"):
        out["RmtInf_Ustrd"] = mt.get("70")
    return out


# ------------------------------------------------------------------ MX -> MT


def mx_to_dict(xml_text: str) -> dict[str, str]:
    """Flatten a pacs.008 document into the shared key space."""
    root = ET.fromstring(xml_text)
    ns = {"n": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
    out: dict[str, str] = {}

    def find(path: str) -> str:
        node = root.find(path, ns) if ns else root.find(path)
        return (node.text or "").strip() if node is not None and node.text else ""

    out["PmtId_EndToEndId"] = find(".//n:PmtId/n:EndToEndId")
    out["PmtId_InstrId"] = find(".//n:PmtId/n:InstrId")
    out["PmtId_UETR"] = find(".//n:PmtId/n:UETR")
    out["IntrBkSttlmDt"] = find(".//n:IntrBkSttlmDt")

    amount_el = root.find(".//n:IntrBkSttlmAmt", ns) if ns else None
    if amount_el is not None:
        out["Amt_IntrBkSttlmAmt"] = (amount_el.text or "").strip()
        ccy = amount_el.get("Ccy")
        if ccy:
            out["Amt_IntrBkSttlmAmt_Ccy"] = ccy
    for key, path in (
        ("Dbtr_Nm", ".//n:Dbtr/n:Nm"),
        ("DbtrAcct_IBAN", ".//n:DbtrAcct//n:IBAN"),
        ("DbtrAgt_BICFI", ".//n:DbtrAgt//n:BICFI"),
        ("Cdtr_Nm", ".//n:Cdtr/n:Nm"),
        ("CdtrAcct_IBAN", ".//n:CdtrAcct//n:IBAN"),
        ("CdtrAgt_BICFI", ".//n:CdtrAgt//n:BICFI"),
        ("ChrgBr", ".//n:ChrgBr"),
        ("RmtInf_Ustrd", ".//n:RmtInf/n:Ustrd"),
    ):
        value = find(path)
        if value:
            out[key] = value
    return {k: v for k, v in out.items() if v}


def mt_logical_terminal(bic: str) -> str:
    """Build the 12-character logical terminal address used in MT blocks 1 and 2.

    The shape is BIC8 + terminal identifier + branch code. An 11-character BIC
    already carries its branch, so appending a branch again produces a 15-character
    field that no MT parser will accept.
    """
    compact = bic.strip().upper()
    bic8 = compact[:8]
    branch = compact[8:11] if len(compact) >= 11 else "XXX"
    return f"{bic8}A{branch}"


def dict_to_mt103(
    values: dict[str, str],
    sender_bic: str | None = None,
    receiver_bic: str | None = None,
) -> str:
    """Render the shared key space as an MT103.

    This is what converts one golden pair into unlimited synthetic ones: sample a
    valid MX message, flatten it, render the MT, and the pair is correct by
    construction.
    """
    sender = sender_bic or values.get("InstgAgt_BICFI") or "BANKDEFF"
    receiver = receiver_bic or values.get("InstdAgt_BICFI") or "BANKFRPP"
    sender_lta = mt_logical_terminal(sender)
    receiver_bic8 = receiver.strip().upper()[:8]
    uetr = values.get("PmtId_UETR") or "00000000-0000-4000-8000-000000000000"

    raw_amount = values.get("Amt_IntrBkSttlmAmt", "0.00")
    mt_amount = (
        f"{float(raw_amount):,.2f}".replace(",", "\u0000")
        .replace(".", ",")
        .replace("\u0000", "")
    )

    date = values.get("IntrBkSttlmDt") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        ymd = datetime.strptime(date, "%Y-%m-%d").strftime("%y%m%d")
    except ValueError:
        ymd = datetime.now(timezone.utc).strftime("%y%m%d")

    charge = CHARGE_BEARER_MX_TO_MT.get(values.get("ChrgBr", "SHAR"), "SHA")

    lines = [
        f"{{1:F01{sender_lta}0000000000}}{{2:I103{receiver_bic8}XXXXN}}"
        f"{{3:{{121:{uetr}}}}}{{4:",
        f":20:{values.get('PmtId_InstrId') or 'NOTPROVIDED'}",
        ":23B:CRED",
        f":32A:{ymd}{values.get('Amt_IntrBkSttlmAmt_Ccy', 'EUR')}{mt_amount}",
    ]
    dbtr_acct = values.get("DbtrAcct_IBAN")
    lines.append(f":50K:/{dbtr_acct}" if dbtr_acct else ":50K:")
    lines.append(values.get("Dbtr_Nm", "UNKNOWN"))
    cdtr_acct = values.get("CdtrAcct_IBAN")
    lines.append(f":59:/{cdtr_acct}" if cdtr_acct else ":59:")
    lines.append(values.get("Cdtr_Nm", "UNKNOWN"))
    if values.get("RmtInf_Ustrd"):
        lines.append(f":70:{values['RmtInf_Ustrd']}")
    lines.append(f":71A:{charge}")
    if values.get("PmtId_EndToEndId"):
        lines.append(f":72:/EToE/{values['PmtId_EndToEndId']}")
    lines.append("-}")
    return "\n".join(lines)


def mx_to_mt103(xml_text: str) -> str:
    """Full conversion: pacs.008 document -> MT103 text."""
    return dict_to_mt103(mx_to_dict(xml_text))


if __name__ == "__main__":
    import sys
    from pathlib import Path

    dual = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/vgi/data/dual")
    mt_text = (dual / "mt" / "payment1.txt").read_text()
    mx_text = (dual / "mx" / "payment1.xml").read_text()

    print("=" * 78)
    print("GOLDEN PAIR: MT103 -> shared key space")
    print("=" * 78)
    from_mt = mt103_to_dict(parse_mt103(mt_text))
    for k, v in sorted(from_mt.items()):
        print(f"  {k:<28} {v}")

    print()
    print("=" * 78)
    print("SAME PAIR: pacs.008 -> shared key space")
    print("=" * 78)
    from_mx = mx_to_dict(mx_text)
    for k, v in sorted(from_mx.items()):
        print(f"  {k:<28} {v}")

    print()
    print("=" * 78)
    print("AGREEMENT — where the two syntaxes disagree")
    print("=" * 78)
    all_keys = sorted(set(from_mt) | set(from_mx))
    agree = disagree = only_mt = only_mx = 0
    for key in all_keys:
        a, b = from_mt.get(key), from_mx.get(key)
        if a and b and a == b:
            agree += 1
        elif a and b:
            disagree += 1
            print(f"  DIFFER  {key:<26} MT={a!r}  MX={b!r}")
        elif a:
            only_mt += 1
            print(f"  MT-only {key:<26} {a!r}")
        else:
            only_mx += 1
            print(f"  MX-only {key:<26} {b!r}")
    print(f"\n  agree={agree}  differ={disagree}  mt-only={only_mt}  mx-only={only_mx}")

    print()
    print("=" * 78)
    print("ROUND TRIP: pacs.008 -> MT103 -> shared key space")
    print("=" * 78)
    regenerated = mt103_to_dict(parse_mt103(mx_to_mt103(mx_text)))
    for key in sorted(set(from_mx) | set(regenerated)):
        a, b = from_mx.get(key, ""), regenerated.get(key, "")
        mark = "ok " if a == b else "DIFF"
        print(f"  {mark} {key:<28} mx={a!r} rt={b!r}")
