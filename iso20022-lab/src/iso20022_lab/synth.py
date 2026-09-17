"""Validated value synthesis, for corpus volume.

Seven real seed value-sets is too thin to measure an exception rate on. The
fixtures under data/ are a seed, not a dataset, and there is no larger public
corpus for the PII reason.

This generates value-sets that are structurally real and lexically valid:

* IBANs carry **correct mod-97 check digits** for their country's real format,
  so the deterministic validator accepts them and the measurement is not
  dominated by values no real document would contain.
* BICs match the ISO 9362 shape for their country prefix.
* Amounts respect the currency's **minor units** (JPY has none), which is the
  gate that correctly rejects `125000.50` JPY.
* Dates, references and remittance text are drawn from realistic pools.

What this does NOT do is invent document *structure* -- layouts and phrasing
still come from the renderer in `corpus.py`, which is the honest part of the
task. What it provides is variation in the values, so a rule that got lucky on
one seed does not look accurate across fifty.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Real national IBAN formats: country, length, bank code length, account length.
IBAN_FORMATS: tuple[tuple[str, int, int, int], ...] = (
    ("DE", 22, 8, 10),
    ("GB", 22, 4, 14),
    ("FR", 27, 10, 11),
    ("NL", 18, 4, 10),
    ("ES", 24, 8, 10),
    ("IT", 27, 10, 12),
)

# Currency -> minor units. JPY has none, which is the trap the minor-unit gate
# catches: 125000.50 JPY is not a representable amount.
CURRENCIES: dict[str, int] = {"EUR": 2, "GBP": 2, "USD": 2, "JPY": 0, "CHF": 2}

# Country -> BIC prefix, so a synthesised BIC is consistent with its IBAN.
BIC_PREFIX: dict[str, str] = {
    "DE": "DEUTDEFF",
    "GB": "BUKBGB22",
    "FR": "BNPAFRPP",
    "NL": "ABNANL2A",
    "ES": "BBVAESMM",
    "IT": "UNCRITMM",
}

COMPANY_NAMES: tuple[str, ...] = (
    "Northwind Trading Ltd",
    "Acme GmbH",
    "Meridian Logistics BV",
    "Blackwood & Sons",
    "Castellane SA",
    "Delta Freight NV",
    "Harbour Supplies Co",
    "Ironbridge Holdings",
    "Juniper Partners LLP",
    "Kestrel Industries",
    "Lakeside Chemicals AG",
    "Montrose Marine Ltd",
    "Nordkap Energi AS",
    "Orchid Pharma BV",
    "Pinewood Textiles",
    "Quarry Ridge Minerals",
    "Ravenscourt Metals",
    "Stonebridge Utilities",
    "Tamarind Foods Ltd",
    "Umbra Consulting SRL",
    "Vantage Aerospace SA",
    "Westport Fisheries",
)

PERSON_NAMES: tuple[str, ...] = (
    "A. Kowalski",
    "B. Ferreira",
    "C. Nwosu",
    "D. Lindqvist",
    "E. Marchetti",
    "F. Okafor",
    "G. Petrov",
    "H. Ramirez",
    "I. Takahashi",
    "J. Van Der Berg",
)

REMITTANCE_PHRASES: tuple[str, ...] = (
    "Invoice {n} settlement",
    "Payment for services ref {n}",
    "Balance of account {n}",
    "Consulting fees {n}",
    "Freight charges {n}",
    "Quarterly maintenance {n}",
    "Goods supplied per order {n}",
    "Reimbursement claim {n}",
)


def iban_check_digits(country: str, bban: str) -> str:
    """Compute ISO 13616 mod-97 check digits for a BBAN.

    Rearranges to `BBAN + country + "00"`, maps letters to 10-35, and takes
    98 - (value mod 97). This is the same arithmetic the validator reverses, so
    a synthesised IBAN is one the validator will accept.
    """
    rearranged = bban + country + "00"
    digits = "".join(str(ord(ch) - 55) if ch.isalpha() else ch for ch in rearranged)
    return f"{98 - (int(digits) % 97):02d}"


def make_iban(rng: random.Random) -> tuple[str, str]:
    """Return (iban, country) with a valid check digit pair."""
    country, _length, bank_len, acct_len = rng.choice(IBAN_FORMATS)
    bank = "".join(str(rng.randint(0, 9)) for _ in range(bank_len))
    acct = "".join(str(rng.randint(0, 9)) for _ in range(acct_len))
    bban = bank + acct
    return f"{country}{iban_check_digits(country, bban)}{bban}", country


def make_bic(rng: random.Random, country: str) -> str:
    """Return a BIC consistent with the country, 8 or 11 characters."""
    base = BIC_PREFIX.get(country, "DEUTDEFF")
    if rng.random() < 0.5:
        return base
    return base + "".join(str(rng.randint(0, 9)) for _ in range(3))


def make_amount(
    rng: random.Random, currency: str, low: float = 50.0, high: float = 500_000.0
) -> str:
    """Return an amount respecting the currency's minor units."""
    units = CURRENCIES[currency]
    value = rng.uniform(low, high)
    if units == 0:
        return f"{int(value)}"
    return f"{value:.{units}f}"


def make_date(rng: random.Random) -> str:
    """Return an ISO date in a plausible forward window."""
    year = rng.choice((2026, 2027))
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"


def make_reference(rng: random.Random) -> str:
    """Return a reference in one of several real-world shapes."""
    return rng.choice(
        (
            f"E2E{rng.randint(100000, 999999)}",
            f"REF-{rng.randint(10000, 99999)}",
            f"PMT{rng.randint(20260000, 20269999)}",
            f"{rng.randint(1000, 9999)}/{rng.randint(100, 999)}",
        )
    )


def make_remittance(rng: random.Random) -> str:
    phrase = rng.choice(REMITTANCE_PHRASES)
    return phrase.format(n=rng.randint(10000, 99999))


@dataclass(frozen=True)
class ValueSet:
    """One canonical value-set, ready to render."""

    values: dict[str, str]
    currency: str
    country: str


def generate_values(
    rng: random.Random,
    currency: str | None = None,
    with_jpy: bool = False,
) -> ValueSet:
    """Generate one internally consistent, validator-clean value-set."""
    ccy = currency or rng.choice(("EUR", "GBP", "USD", "CHF", "JPY" if with_jpy else "EUR"))
    cdtr_iban, cdtr_country = make_iban(rng)
    dbtr_iban, dbtr_country = make_iban(rng)
    values: dict[str, str] = {
        "Cdtr_Nm": rng.choice(COMPANY_NAMES + PERSON_NAMES),
        "CdtrAcct_IBAN": cdtr_iban,
        "CdtrAgt_BICFI": make_bic(rng, cdtr_country),
        "Dbtr_Nm": rng.choice(COMPANY_NAMES),
        "DbtrAcct_IBAN": dbtr_iban,
        "Amt_InstdAmt": make_amount(rng, ccy),
        "Amt_InstdAmt_Ccy": ccy,
        "PmtId_EndToEndId": make_reference(rng),
        "ReqdExctnDt_Dt": make_date(rng),
        "RmtInf_Ustrd": make_remittance(rng),
    }
    _ = dbtr_country
    return ValueSet(values=values, currency=ccy, country=cdtr_country)


def generate_many(
    count: int,
    seed: int = 0,
    currency_mix: tuple[str, ...] = ("EUR", "GBP", "USD", "CHF"),
) -> list[dict[str, str]]:
    """Generate `count` value-sets across a realistic currency mix."""
    rng = random.Random(seed)
    out: list[dict[str, str]] = []
    for i in range(count):
        ccy = currency_mix[i % len(currency_mix)]
        out.append(generate_values(rng, currency=ccy).values)
    return out


def main() -> int:
    from iso20022_lab.validators import iban_valid, bic_valid, validate_field

    rng = random.Random(11)
    generated = [generate_values(rng).values for _ in range(200)]

    bad_iban = [v["CdtrAcct_IBAN"] for v in generated if not iban_valid(v["CdtrAcct_IBAN"])]
    bad_bic = [v["CdtrAgt_BICFI"] for v in generated if not bic_valid(v["CdtrAgt_BICFI"])]
    findings: list[str] = []
    for values in generated:
        for key, value in values.items():
            for finding in validate_field(key, value):
                findings.append(f"{key}={value}: {finding.detail}")

    print(f"generated {len(generated)} value-sets")
    print(f"  invalid IBANs : {len(bad_iban)}")
    print(f"  invalid BICs  : {len(bad_bic)}")
    print(f"  validator     : {len(findings)} findings")
    for line in findings[:8]:
        print(f"    {line}")
    print()
    print("sample:")
    for key, value in generated[0].items():
        print(f"  {key:<20} {value}")
    return 0 if not bad_iban and not bad_bic and not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
