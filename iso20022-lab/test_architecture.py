"""
The architecture claim, tested.

PROCESS.md asserts: element order comes from the XSD, so slot dict key order is
irrelevant and a shuffled dict produces byte-identical XML. If this test fails,
the central design argument is wrong and the approach collapses back into
"ask a model to write XML".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from iso20022_lab.serialize import (
    MessageBuilder,
    Slots,
    Value,
    describe,
    validate,
)
from iso20022_lab.xsd_introspect import XSDModel

SCHEMA = Path(__file__).parent / "schemas" / "pain.001.001.09.xsd"


def slots_ordered() -> Slots:
    return {
        "CstmrCdtTrfInitn": {
            "GrpHdr": {
                "MsgId": "MSG-2026-0916-001",
                "CreDtTm": "2026-09-16T10:00:00Z",
                "NbOfTxs": "1",
                "InitgPty": {"Nm": "Northwind Trading Limited"},
            },
            "PmtInf": [
                {
                    "PmtInfId": "PMT-001",
                    "PmtMtd": "TRF",
                    "ReqdExctnDt": {"Dt": "2026-09-20"},
                    "Dbtr": {"Nm": "Northwind Trading Limited"},
                    "DbtrAcct": {"Id": {"IBAN": "GB33BUKB20201555555555"}},
                    "DbtrAgt": {"FinInstnId": {"BICFI": "BUKBGB22"}},
                    "CdtTrfTxInf": [
                        {
                            "PmtId": {"EndToEndId": "E2E-001"},
                            "Amt": {"InstdAmt": {"_value": "1250.00", "Ccy": "EUR"}},
                            "CdtrAgt": {"FinInstnId": {"BICFI": "DEUTDEFF"}},
                            "Cdtr": {"Nm": "ACME GmbH"},
                            "CdtrAcct": {"Id": {"IBAN": "DE89370400440532013000"}},
                            "ChrgBr": "SHAR",
                        }
                    ],
                }
            ],
        }
    }


def reverse_deep(obj: Slots) -> Slots:
    """Rebuild a Slots tree with every mapping's keys in reverse order."""
    out: dict[str, Value | Slots | Sequence[Value | Slots]] = {}
    for key, value in reversed(list(obj.items())):
        if isinstance(value, Mapping):
            out[key] = reverse_deep(value)
        elif isinstance(value, list):
            out[key] = [
                reverse_deep(item) if isinstance(item, Mapping) else item for item in value
            ]
        else:
            out[key] = value
    return out


def main() -> int:
    model = XSDModel(SCHEMA)
    builder = MessageBuilder(model)

    print("=" * 78)
    print("SCHEMA SUMMARY")
    print("=" * 78)
    print(describe(model))

    a = builder.build(slots_ordered())
    b = builder.build(reverse_deep(slots_ordered()))

    print()
    print("=" * 78)
    print("TEST A - shuffled slot key order produces identical XML")
    print("=" * 78)
    print(f"build A ok : {a.ok}   errors={a.errors}")
    print(f"build B ok : {b.ok}   errors={b.errors}")
    identical = a.xml == b.xml
    print(f"byte-identical: {identical}")
    if not identical:
        print("\n--- A ---")
        print(a.xml)
        print("--- B ---")
        print(b.xml)

    print()
    print("=" * 78)
    print("TEST B - validates against the real ISO 20022 XSD")
    print("=" * 78)
    if a.xml:
        ok, errs = validate(a.xml, SCHEMA)
        print(f"XSD valid: {ok}")
        for err in errs[:10]:
            print(f"  {err}")

    print()
    print("=" * 78)
    print("TEST C - bad input is caught, and names the failing path")
    print("=" * 78)
    bad_raw = {
        "CstmrCdtTrfInitn": {
            "GrpHdr": {
                "CreDtTm": "2026-09-16T10:00:00Z",
                "NbOfTxs": "1",
                "InitgPty": {"Nm": "Northwind Trading Limited"},
                "BogusField": "x",
            },
            "PmtInf": [
                {
                    "PmtInfId": "PMT-001",
                    "PmtMtd": "TRF",
                    "ReqdExctnDt": {"Dt": "2026-09-20"},
                    "Dbtr": {"Nm": "Northwind Trading Limited"},
                    "DbtrAcct": {"Id": {"IBAN": "GB33BUKB20201555555555"}},
                    "DbtrAgt": {"FinInstnId": {"BICFI": "BUKBGB22"}},
                    "CdtTrfTxInf": [
                        {
                            "PmtId": {"EndToEndId": "E2E-001"},
                            "Amt": {"InstdAmt": {"_value": "1250.00", "Ccy": "EUR"}},
                            "CdtrAgt": {"FinInstnId": {"BICFI": "DEUTDEFF"}},
                            "Cdtr": {"Nm": "ACME GmbH"},
                            "CdtrAcct": {"Id": {"IBAN": "DE89370400440532013000"}},
                            "ChrgBr": "NOPE",
                        }
                    ],
                }
            ],
        }
    }
    bad: Slots = bad_raw
    c = builder.build(bad)
    print(f"ok: {c.ok}")
    for err in c.errors:
        print(f"  error: {err}")
    for warn in c.warnings:
        print(f"  warn : {warn}")

    verdict = identical and a.ok
    print()
    print("=" * 78)
    print(f"ARCHITECTURE CLAIM: {'HOLDS' if verdict else 'FAILS'}")
    print("=" * 78)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
