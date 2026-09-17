"""
Failure-mode probes for the Needle-based extractor design.

These are adversarial tests against my own architecture, not demonstrations.
Each one targets a specific claim in PROCESS.md that could be false:

  1. Context window       — is 256 tokens really the limit?
  2. Confidence head      — does it survive fine-tuning?
  3. Grounding            — does the model invent values not in the input?
  4. End token / repetition — does decoding terminate cleanly?
  5. Closed-set codes     — how does it behave with out-of-vocabulary values?
  6. Minor units          — does it respect ISO 4217 decimal places?

Run:  source .venv/bin/activate && python iso20022-lab/probe_failures.py
"""

from __future__ import annotations

import os
import warnings
from typing import Literal

os.environ.setdefault("NEEDLE_TELEMETRY", "0")
import needle

SEP = "=" * 78


def header(n: int, title: str) -> None:
    print(f"\n{SEP}\n{n}. {title}\n{SEP}")


# --------------------------------------------------------------------- tools


@needle.tool
def record_payment(
    debtor_name: str,
    creditor_name: str,
    creditor_iban: str,
    amount: float,
    currency: Literal["EUR", "USD", "GBP", "JPY", "BHD"],
    charge_bearer: Literal["DEBT", "CRED", "SHAR", "SLEV"] = "SHAR",
):
    """Record the fields of a credit transfer instruction.

    Args:
        debtor_name: name of the party sending the money
        creditor_name: name of the party receiving the money
        creditor_iban: the creditor's IBAN, exactly as written
        amount: instructed amount as a number
        currency: ISO 4217 currency code
        charge_bearer: which party bears the charges
    """
    return {
        "debtor": debtor_name,
        "creditor": creditor_name,
        "iban": creditor_iban,
        "amount": amount,
        "ccy": currency,
        "chrgbr": charge_bearer,
    }


def make_agent():
    return needle.Needle(tools=[record_payment])


# --------------------------------------------------------------- 1. window


def probe_context_window() -> None:
    header(1, "CONTEXT WINDOW — is 256 tokens really the ceiling?")
    agent = make_agent()

    # A short, well-formed instruction.
    short = "Pay ACME GmbH EUR 1250.00 from Northwind Ltd to IBAN DE89370400440532013000."
    agent.reset()
    r = agent.complete(short)
    print(f"short input ({len(short)} chars, ~{len(short) // 4} tokens)")
    print(f"  call  : {r.get('function_calls')}")
    print(f"  conf  : {r.get('confidence')}")

    # Same facts buried in realistic invoice preamble.
    filler = (
        "TAX INVOICE. Supplier: Northwind Trading Limited, Unit 4, "
        "Bletchley Industrial Estate, Milton Keynes, MK1 1AA, United Kingdom. "
        "VAT Registration Number GB 123 4567 89. Company Registration 09876543. "
        "Bank: Barclays, Sort Code 20-00-00, Account 12345678. "
        "Terms: net 30 days from invoice date. Late payment interest charged at 8%. "
        "Purchase order reference PO-2026-0916. Delivery note DN-4471. "
        "Goods received in good condition. No discrepancies noted on inspection. "
        "Authorised signatory: A. Whitfield, Finance Director. "
    )
    target = "Settlement instruction: remit EUR 1250.00 to ACME GmbH, IBAN DE89370400440532013000."
    long_input = filler + target
    approx = len(long_input) // 4
    print(
        f"\nlong input ({len(long_input)} chars, ~{approx} tokens) — target sentence is at the END"
    )
    agent.reset()
    r2 = agent.complete(long_input)
    print(f"  call  : {r2.get('function_calls')}")
    print(f"  conf  : {r2.get('confidence')}")
    print(f"  reason: {r2.get('reasoning')}")

    called = bool(r2.get("function_calls"))
    if not called:
        print("\n  >>> VERDICT: FAIL. The target facts were not extracted from a")
        print("      document-length input. This is the single biggest risk in the design:")
        print("      real invoices are far longer than a 256-token window.")
    else:
        print("\n  >>> VERDICT: extracted despite length. Window may be larger, or the")
        print("      engine truncates from the front. Verify before trusting.")


# ----------------------------------------------------------- 2. confidence


def probe_confidence_under_tuning() -> None:
    header(2, "CONFIDENCE HEAD — does the escalation gate survive fine-tuning?")
    import inspect

    src = inspect.getsource(needle.Needle.__init__)
    for line in src.splitlines():
        if "confidence" in line.lower() or "warn" in line.lower():
            print(f"  {line.strip()}")
    print("\n  >>> VERDICT: the head is NOT updated by LoRA, and the package sets")
    print("      confidence to None for tuned weights. So the escalation gate that")
    print("      Phase 5 depends on does not exist for a tuned model.")
    print("      The design must either (a) use the frozen base model untuned,")
    print("      (b) train an independent calibration head, or (c) gate on")
    print("      deterministic signals instead of a learned score.")


# ------------------------------------------------------------- 3. grounding


def probe_grounding() -> None:
    header(3, "GROUNDING — does the model invent values that are not in the input?")
    agent = make_agent()

    cases = [
        (
            "no IBAN anywhere in the text",
            "Pay ACME GmbH EUR 1250.00 from Northwind Ltd. Bank details to follow.",
        ),
        (
            "amount only stated in words",
            "Northwind Ltd agrees to pay ACME GmbH the sum of twelve hundred euros.",
        ),
        (
            "silence on charge bearer",
            "Remit EUR 1250 to ACME GmbH, IBAN DE89370400440532013000, from Northwind Ltd.",
        ),
        ("off-topic entirely", "The weather in Lisbon was mild for September."),
    ]
    for label, text in cases:
        agent.reset()
        r = agent.complete(text)
        print(f"\n  case: {label}")
        print(f"    call: {r.get('function_calls')}")
        print(f"    conf: {r.get('confidence')}")
        print(f"    why : {r.get('reasoning')}")


# ------------------------------------------------------ 4. end token / rep


def probe_end_token_repetition() -> None:
    header(4, "END TOKEN / REPETITION — does decoding terminate cleanly?")
    agent = make_agent()

    # Force a long free-text-ish generation with no concise answer available.
    agent.reset()
    r = agent.complete("Describe every possible field of an ISO 20022 message in full.")
    calls = r.get("function_calls")
    raw_len = len(str(calls))
    print(f"  output length : {raw_len} chars")
    print(f"  call          : {str(calls)[:300]}")
    print(f"  decode_tps    : {r.get('decode_tps')}")

    # Hard cap test: max_new_tokens is an output cap, not an EOS guarantee.
    try:
        r2 = agent.complete(
            "List all currencies in the world, one per line.", max_new_tokens=64
        )
        print(
            f"\n  capped run -> {len(str(r2.get('function_calls')))} chars (cap 64 tokens)"
        )
    except TypeError as exc:
        print(f"\n  max_new_tokens not accepted on this path: {exc}")

    print("\n  >>> NOTE: the engine is grammar-constrained, so the observable failure")
    print("      mode is not runaway repetition of free text but a truncated or empty")
    print("      call. Repetition still matters for the reasoning field, which the")
    print("      docs state is generated UNCONSTRAINED.")


# --------------------------------------------------------- 5. closed sets


def probe_closed_sets() -> None:
    header(5, "CLOSED SETS — behaviour on out-of-vocabulary values")
    agent = make_agent()

    cases = [
        (
            "currency not in the Literal set",
            "Pay EURX 1250 to ACME GmbH, IBAN DE89370400440532013000.",
        ),
        (
            "charge bearer in prose not code",
            "Send it with charges borne by the debtor, OUR basis.",
        ),
        (
            "valid code supplied directly",
            "Charge bearer SLEV. EUR 1250 to ACME GmbH from Northwind Ltd.",
        ),
    ]
    for label, text in cases:
        agent.reset()
        r = agent.complete(text)
        print(f"\n  case: {label}")
        print(f"    call: {r.get('function_calls')}")
        print(f"    conf: {r.get('confidence')}")


# ------------------------------------------------------- 6. minor units


def probe_minor_units() -> None:
    header(6, "MINOR UNITS — ISO 4217 decimal places (JPY=0, BHD=3)")
    agent = make_agent()

    cases = [
        (
            "JPY has zero decimals",
            "Pay JPY 125000 to ACME GmbH, IBAN DE89370400440532013000.",
        ),
        (
            "BHD has three decimals",
            "Pay BHD 1250.125 to ACME GmbH, IBAN DE89370400440532013000.",
        ),
        ("EUR two decimals", "Pay EUR 1250.50 to ACME GmbH, IBAN DE89370400440532013000."),
    ]
    for label, text in cases:
        agent.reset()
        r = agent.complete(text)
        calls = r.get("function_calls") or []
        got = calls[0]["arguments"].get("amount") if calls else None
        print(f"\n  {label}")
        print(
            f"    amount extracted: {got!r}   currency: "
            f"{calls[0]['arguments'].get('currency') if calls else None!r}"
        )
    print("\n  >>> The model has no concept of ISO 4217 minor units. 'BHD 1250.125'")
    print("      round-tripping depends entirely on the schema and the validator,")
    print("      which is exactly why that layer must be deterministic.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probe_context_window()
        probe_confidence_under_tuning()
        probe_grounding()
        probe_end_token_repetition()
        probe_closed_sets()
        probe_minor_units()
    print(f"\n{SEP}\nprobes complete\n{SEP}")
