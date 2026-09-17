"""
Deterministic validators: the precision backbone of the distillation pipeline.

The design principle that makes a cheap teacher sufficient:

    the teacher supplies COVERAGE, the validators supply PRECISION.

A weak teacher that produces many candidate labels is fine, because every label
is checked here before it becomes training data. A single wrong IBAN is rejected
mechanically, and it is rejected for a reason a human can read and audit. This is
also the only layer that can be trusted at inference time, since the probes in
PROCESS.md section 2.8 showed the model itself has no concept of minor units or
checksums.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ISO 4217 minor units. Anything not listed is assumed to have 2 decimals, which
# is the majority case, but the exceptions below are the ones that silently
# corrupt amounts if you get them wrong.
MINOR_UNITS: dict[str, int] = {
    # zero-decimal currencies
    "BIF": 0,
    "CLP": 0,
    "DJF": 0,
    "GNF": 0,
    "ISK": 0,
    "JPY": 0,
    "KMF": 0,
    "KRW": 0,
    "PYG": 0,
    "RWF": 0,
    "UGX": 0,
    "UYI": 0,
    "VND": 0,
    "VUV": 0,
    "XAF": 0,
    "XOF": 0,
    "XPF": 0,
    # three-decimal currencies
    "BHD": 3,
    "IQD": 3,
    "JOD": 3,
    "KWD": 3,
    "LYD": 3,
    "OMR": 3,
    "TND": 3,
    # four-decimal
    "CLF": 4,
    "UYW": 4,
}

CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
BIC_RE = re.compile(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$")

# ISO 3166-1 alpha-2. ISO 9362 puts one of these in positions 5-6 of a BIC, which
# `BIC_RE` alone cannot check -- it validates shape only.
#
# This is not pedantry. Ordinary English words match the BIC shape, and several
# of them land on REAL country codes, so no amount of shape checking catches
# them:
#
#     attached     -> ATTACHED     (CH, Switzerland)
#     beneficiary  -> BENEFICIARY  (FI, Finland)
#     instruction  -> INSTRUCTION  (RU, Russia)
#     transferred  -> TRANSFERRED  (SF, not a country)
#     received     -> RECEIVED     (IV, not a country)
#
# When a probe uses BIC shape as evidence that a text layer is sound, a false
# VALID is the dangerous direction: it raises the trust score and manufactures
# confidence in a document where nothing was actually verified.
ISO_3166_ALPHA2: frozenset[str] = frozenset(
    """
    AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ BA BB BD BE BF BG BH BI BJ
    BL BM BN BO BQ BR BS BT BV BW BY BZ CA CC CD CF CG CH CI CK CL CM CN CO CR
    CU CV CW CX CY CZ DE DJ DK DM DO DZ EC EE EG EH ER ES ET FI FJ FK FM FO FR
    GA GB GD GE GF GG GH GI GL GM GN GP GQ GR GS GT GU GW GY HK HM HN HR HT HU
    ID IE IL IM IN IO IQ IR IS IT JE JM JO JP KE KG KH KI KM KN KP KR KW KY KZ
    LA LB LC LI LK LR LS LT LU LV LY MA MC MD ME MF MG MH MK ML MM MN MO MP MQ
    MR MS MT MU MV MW MX MY MZ NA NC NE NF NG NI NL NO NP NR NU NZ OM PA PE PF
    PG PH PK PL PM PN PR PS PT PW PY QA RE RO RS RU RW SA SB SC SD SE SG SH SI
    SJ SK SL SM SN SO SR SS ST SV SX SY SZ TC TD TF TG TH TJ TK TL TM TN TO TR
    TT TV TW TZ UA UG UM US UY UZ VA VC VE VG VI VN VU WF WS YE YT ZA ZM ZW
    """.split()
)
UETR_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)
LEI_RE = re.compile(r"^[A-Z0-9]{18}[0-9]{2}$")


@dataclass
class Finding:
    field: str
    value: str
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.field}={self.value!r} violates {self.rule}: {self.detail}"


def iban_valid(iban: str) -> bool:
    """ISO 13616 mod-97 check (letters -> 10..35, remainder must be 1)."""
    compact = re.sub(r"\s+", "", iban).upper()
    if not 5 <= len(compact) <= 34:
        return False
    if not compact[:2].isalpha() or not compact[2:4].isdigit():
        return False
    if not compact.isalnum():
        return False
    rearranged = compact[4:] + compact[:4]
    digits = "".join(str(int(ch, 36)) if ch.isalpha() else ch for ch in rearranged)
    try:
        return int(digits) % 97 == 1
    except ValueError:
        return False


def bic_valid(bic: str) -> bool:
    """Shape only, per ISO 9362. Does NOT check the country code.

    Deliberately left loose because it also validates extracted fields, where a
    downstream gate or a human sees the value. Use `bic_country_valid` when the
    check is evidence rather than a filter -- see the note on ISO_3166_ALPHA2.
    """
    return bool(BIC_RE.match(bic.strip().upper()))


def bic_country_valid(bic: str) -> bool:
    """A BIC whose country code (positions 5-6) is a real ISO 3166 country.

    This check was introduced to reject prose that matches the BIC shape, and its
    docstring used to claim that `attached`, `beneficiary` and `instruction` fail
    it. **That claim was false, and all three pass.** Positions 5-6 of those
    words are `CH` (Switzerland), `FI` (Finland) and `RU` (Russia) -- all real
    country codes.

    The correction matters beyond a comment, because the false version described a
    guarantee the pipeline does not have. What actually rejects prose is C8's
    context anchoring in `acceptance.py`: a BIC is accepted only when the source
    anchors it to a BIC label. Shape plus a real country code is not evidence, at
    either end of the pipeline.

    Measured on real PP-OCRv6 output, this check also cannot detect a *corrupted*
    BIC, because recognition damage frequently lands on another real country:
    `BNPAFRPP704` read as `BNPAERPP704` is a valid BIC for Eritrea, and
    `ABNANL2A836` read as `ARNANI2A836` is a valid BIC for Nicaragua. The check
    discriminates between *prose* and *a value*, not between a value and a value
    that was misread. See PROCESS.md 18.12.

    Kept because it does reject some prose, cheaply and deterministically. It is
    not evidence that a BIC is correct, and nothing should treat it as such.
    """
    compact = bic.strip().upper()
    if len(compact) not in (8, 11):
        return False
    if not BIC_RE.match(compact):
        return False
    return compact[4:6] in ISO_3166_ALPHA2


def uetr_valid(uetr: str) -> bool:
    """UETR is a v4 UUID, per the SWIFT gpi specification."""
    return bool(UETR_RE.match(uetr.strip()))


def lei_valid(lei: str) -> bool:
    return bool(LEI_RE.match(lei.strip().upper()))


def currency_valid(code: str) -> bool:
    return bool(CURRENCY_RE.match(code.strip().upper()))


def amount_matches_minor_units(amount: str | float | int, currency: str) -> bool:
    """True when the amount's decimal places match the currency's minor units.

    This is the check that catches 'JPY 125000.50' and 'BHD 1250.12'. The model
    cannot do it -- the probes showed it fails every minor-unit case.
    """
    expected = MINOR_UNITS.get(currency.strip().upper(), 2)
    text = str(amount).strip()
    if "." in text:
        decimals = len(text.split(".", 1)[1])
    else:
        decimals = 0
    return decimals == expected


def validate_field(field_name: str, value: object) -> list[Finding]:
    """Apply every rule that could plausibly apply to a named field.

    Field names are matched on the ISO 20022 element name, so `IBAN`, `BICFI`
    and `UETR` hit their rules directly while unrelated fields pass through.
    """
    findings: list[Finding] = []
    name = field_name.upper()
    text = str(value)

    if "IBAN" in name:
        if not iban_valid(text):
            findings.append(
                Finding(field_name, text, "IBAN-mod97", "checksum or structure failed")
            )
    if "BIC" in name:
        if not bic_valid(text):
            findings.append(
                Finding(field_name, text, "BIC-format", "not 8 or 11 chars in BIC shape")
            )
    if "UETR" in name:
        if not uetr_valid(text):
            findings.append(Finding(field_name, text, "UETR-uuid4", "not a v4 UUID"))
    if "LEI" in name:
        if not lei_valid(text):
            findings.append(Finding(field_name, text, "LEI-format", "18 alnum + 2 digits"))
    if name in {"CCY", "CURRENCY"} or name.endswith("CCY"):
        if not currency_valid(text):
            findings.append(
                Finding(field_name, text, "ISO4217-shape", "not a 3-letter code")
            )
    return findings
