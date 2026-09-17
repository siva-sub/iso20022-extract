"""Cross-checking unchecksummed fields against the payment's own routing data.

THE PROBLEM THIS SOLVES

An extracted BIC cannot be verified by looking at the BIC. That is the finding
PROCESS.md 18.3, 18.11 and 18.12 all arrive at from different directions, and
18.12 measures it: of 9 silently-wrong fields in a recognition run, **all 9 were
`CdtrAgt_BICFI`, and 8 of 9 passed both the shape check and the country check.**

    BNPAFRPP704  read as  BNPAERPP704      FR (France)      -> ER (Eritrea)
    ABNANL2A836  read as  ARNANI2A836      NL (Netherlands) -> NI (Nicaragua)
    DEUTDEFF349  read as  DEUTDEEE349      DE -> DE  (country survived intact)

`ER` and `NI` are real ISO 3166 countries. So a corrupted BIC is frequently a
*valid BIC for a different country*, and no check that inspects the field alone
can tell. Meanwhile, under identical recognition damage, the IBAN field produced
**236 errors and 0 silent ones** -- because mod-97 catches every single-character
change.

THE ASYMMETRY IS THE TOOL

Those two facts combine into something better than either check alone:

    The IBAN is checksummed, so its country code is trustworthy.
    The BIC is not, so its country code is only a claim.
    They describe the SAME party.
    Therefore the trustworthy one can test the untrustworthy one.

That is a cross-field check, and it needs no recogniser improvement, no directory
download and no model. It is deterministic and it catches exactly the class that
was getting through:

    CdtrAcct_IBAN = FR76...   (mod-97 verified)
    CdtrAgt_BICFI = BNPAERPP  (Eritrea)
    -> the account is French, the bank is claimed to be Eritrean: reject

WHAT THIS MODULE DOES NOT DO, AND WHY THAT IS STATED PLAINLY

It cannot verify the *bank code* half of a BIC without a bank directory. A BIC is
four letters of institution, two of country, two of location. Mapping
`DEUTDEFF` <-> German BLZ `37040044` requires the Bundesbank's BLZ/BIC table or
an equivalent, and this project has no such data and will not invent one.

So `RoutingDirectory` is a Protocol with three implementations:

- `NoDirectory` -- the default. Country agreement only. Says so in every report.
- `ObservedDirectory` -- learns IBAN->BIC pairs from documents already confirmed
  correct, which is what a payment shop accumulates in its own master data. This
  is the one a production deployment would use, and it is honest about being
  built from its own history rather than from an authoritative registry.
- a caller-supplied real directory, by implementing the Protocol.

A `Verify` result distinguishes **PASSED** from **UNVERIFIABLE**. A missing
directory produces UNVERIFIABLE, never PASSED -- because "I could not check" and
"I checked and it is fine" are different statements, and collapsing them is how a
check becomes decorative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from iso20022_lab.validators import bic_country_valid, bic_valid, iban_valid

# Which agent BIC belongs with which account. In pain.001 the creditor's agent is
# the bank holding the creditor's account, and the debtor's agent holds the
# debtor's. Keeping the pairing explicit stops this drifting into "compare the
# BIC against whichever IBAN is handy", which would pass on a cross-party mix-up
# -- a wrong-recipient payment, the one error nothing downstream catches.
AGENT_ACCOUNT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("CdtrAgt_BICFI", "CdtrAcct_IBAN", "creditor"),
    ("DbtrAgt_BICFI", "DbtrAcct_IBAN", "debtor"),
)


class Verdict(str, Enum):
    """Outcome of one comparison.

    UNVERIFIABLE is deliberately not folded into PASSED. A BIC with no
    counterparty IBAN, or one issued in a country with no bank-directory entry,
    has not been checked -- and a report that calls that "pass" is worse than one
    that admits the gap, because it manufactures confidence.
    """

    PASSED = "passed"
    UNVERIFIABLE = "unverifiable"
    FAILED = "failed"


@dataclass
class Finding:
    """One cross-check result, with the evidence that produced it."""

    check: str
    verdict: Verdict
    detail: str
    party: str = ""
    bic: str = ""
    iban: str = ""

    @property
    def blocking(self) -> bool:
        return self.verdict is Verdict.FAILED

    def line(self) -> str:
        mark = {"passed": "ok  ", "unverifiable": "??  ", "failed": "FAIL"}[
            self.verdict.value
        ]
        return f"  [{mark}] {self.check}: {self.detail}"


def iban_country(iban: str) -> str:
    """The ISO 3166 alpha-2 country prefix of an IBAN, or ""."""
    compact = "".join(iban.split()).upper()
    return compact[:2] if len(compact) >= 2 and compact[:2].isalpha() else ""


def bic_country(bic: str) -> str:
    """The ISO 3166 alpha-2 country of a BIC, or "".

    Positions 5-6, per ISO 9362. Returns "" rather than a guess when the value is
    too short to carry one.
    """
    compact = "".join(bic.split()).upper()
    return compact[4:6] if len(compact) >= 6 else ""


def bic_bank_code(bic: str) -> str:
    """The institution prefix of a BIC: positions 1-4.

    Not the same thing as an IBAN's bank code. The two are related by a national
    directory, not by any rule this module can apply -- see the module docstring.
    """
    compact = "".join(bic.split()).upper()
    return compact[:4] if len(compact) >= 4 else ""


class RoutingDirectory(Protocol):
    """A source of IBAN->BIC correspondences.

    Deliberately narrow. A payment shop's master data answers one question --
    "which bank holds this account?" -- and that is all this needs to ask.
    """

    name: str

    def bic_for_iban(self, iban: str) -> str | None:
        """The BIC expected to hold `iban`, or None when unknown."""
        ...


class NoDirectory:
    """The default: no bank directory, so only country agreement is checkable.

    Named rather than left as `None` so that every report can say *why* a BIC went
    unverified. "No directory configured" is a deployment fact an operator can
    act on; a bare null is not.
    """

    name = "none"

    def bic_for_iban(self, iban: str) -> str | None:
        return None


@dataclass
class ObservedDirectory:
    """A directory learned from documents that were already confirmed correct.

    This is what a payment operation actually accumulates: every payment that a
    human cleared establishes that a given IBAN is held at a given BIC. It is not
    an authoritative registry and does not claim to be -- but it is derived from
    settled facts rather than from pattern matching, which is more than the BIC
    itself can offer.

    Deliberately does not learn from *predictions*. A directory built from a
    model's own output would rediscover the model's errors as routing rules, and
    the check would then confirm them.
    """

    name = "observed"
    pairs: dict[str, str] = field(default_factory=dict)

    def observe(self, iban: str, bic: str) -> None:
        """Record a correspondence that was confirmed by a human or a settlement."""
        key = _iban_key(iban)
        value = "".join(bic.split()).upper()
        if key and value:
            self.pairs[key] = value

    def bic_for_iban(self, iban: str) -> str | None:
        return self.pairs.get(_iban_key(iban))

    def __len__(self) -> int:
        return len(self.pairs)


def _iban_key(iban: str) -> str:
    """Directory key: the compact uppercase IBAN."""
    return "".join(iban.split()).upper()


def check_country_agreement(
    bic: str, iban: str, party: str, directory: RoutingDirectory
) -> Finding:
    """Does the bank's country match the account's country?

    The IBAN's country is trustworthy because mod-97 was checked first: a
    corrupted IBAN fails its checksum, so anything reaching here with a valid
    checksum has a country prefix that was actually read correctly. The BIC
    carries no such guarantee, which is what makes this comparison informative
    rather than circular.
    """
    bic_cc = bic_country(bic)
    iban_cc = iban_country(iban)

    if not bic_valid(bic):
        return Finding(
            check="bic_shape",
            verdict=Verdict.FAILED,
            detail=f"{bic!r} is not a well-formed BIC",
            party=party,
            bic=bic,
            iban=iban,
        )
    if not iban_valid(iban):
        # A bad IBAN is its own failure, reported elsewhere by the validators. It
        # cannot be used as the reference here, so this returns UNVERIFIABLE
        # rather than pretending to a comparison it cannot support.
        return Finding(
            check="bic_country_matches_iban",
            verdict=Verdict.UNVERIFIABLE,
            detail=(
                f"{party} IBAN {iban!r} does not pass mod-97, so it cannot serve "
                "as the reference country"
            ),
            party=party,
            bic=bic,
            iban=iban,
        )
    if not bic_country_valid(bic):
        return Finding(
            check="bic_country_matches_iban",
            verdict=Verdict.FAILED,
            detail=(
                f"{party} BIC {bic!r} has country {bic_cc!r}, which is not an "
                "assigned ISO 3166 code"
            ),
            party=party,
            bic=bic,
            iban=iban,
        )
    if bic_cc != iban_cc:
        return Finding(
            check="bic_country_matches_iban",
            verdict=Verdict.FAILED,
            detail=(
                f"{party} account is in {iban_cc} ({iban[:6]}...) but the agent "
                f"BIC {bic} is a {bic_cc} institution -- one of the two was "
                "misread, and only the account is checksum-protected"
            ),
            party=party,
            bic=bic,
            iban=iban,
        )
    return Finding(
        check="bic_country_matches_iban",
        verdict=Verdict.PASSED,
        detail=f"{party} BIC country {bic_cc} agrees with the account country",
        party=party,
        bic=bic,
        iban=iban,
    )


def check_directory(
    bic: str, iban: str, party: str, directory: RoutingDirectory
) -> Finding:
    """Does the BIC match the bank the directory says holds this account?

    The strong check. It compares an unchecksummed field against a *known
    correspondence* rather than against a pattern, which is the only way to
    detect a damaged BIC whose damaged form is still a valid BIC.
    """
    expected = directory.bic_for_iban(iban)
    if expected is None:
        return Finding(
            check="bic_matches_directory",
            verdict=Verdict.UNVERIFIABLE,
            detail=(
                f"{party} IBAN is not in the routing directory "
                f"({getattr(directory, 'name', 'unknown')}), so the bank code "
                "cannot be confirmed -- only the country was checkable"
            ),
            party=party,
            bic=bic,
            iban=iban,
        )
    if expected == bic:
        return Finding(
            check="bic_matches_directory",
            verdict=Verdict.PASSED,
            detail=f"{party} BIC matches the directory entry",
            party=party,
            bic=bic,
            iban=iban,
        )
    # Same country, different institution: the exact failure a country check
    # cannot see. `DEUTDEEE349` is a well-formed German BIC for no bank.
    return Finding(
        check="bic_matches_directory",
        verdict=Verdict.FAILED,
        detail=(
            f"{party} account is held at {expected} but the extraction says {bic} "
            "-- same country, different institution"
        ),
        party=party,
        bic=bic,
        iban=iban,
    )


def cross_check(
    payload: dict[str, str],
    directory: RoutingDirectory | None = None,
) -> list[Finding]:
    """Every routing cross-check the payload supports.

    Only pairs where both fields are present are checked. A missing field is a
    completeness problem, reported by the extraction layer, not a routing
    disagreement -- conflating them would blame the routing data for a field the
    recogniser never produced.
    """
    # `is None`, not `or`. `ObservedDirectory` defines `__len__`, so an empty one
    # is FALSY -- and `directory or NoDirectory()` therefore replaced a configured
    # but not-yet-populated directory with "no directory configured". The check
    # still ran and still said unverifiable, so nothing failed loudly; it just
    # attributed the result to the wrong cause, which is the difference between an
    # operator reading "add a directory" and "your directory is empty".
    #
    # More generally: any object that answers `len()` can be falsy, and a
    # truthiness test on a caller-supplied collaborator is a silent substitution
    # waiting to happen.
    resolver: RoutingDirectory = NoDirectory() if directory is None else directory
    findings: list[Finding] = []

    for bic_key, iban_key, party in AGENT_ACCOUNT_PAIRS:
        bic = payload.get(bic_key, "")
        iban = payload.get(iban_key, "")
        if not bic or not iban:
            continue
        findings.append(check_country_agreement(bic, iban, party, resolver))
        # Only worth asking about the bank code when the country already agrees;
        # a country mismatch is the larger finding and two FAILED lines for one
        # cause reads as two problems.
        if findings[-1].verdict is Verdict.PASSED:
            findings.append(check_directory(bic, iban, party, resolver))

    return findings


def summarise(findings: list[Finding]) -> str:
    """One line per finding, plus an explicit count of what was not checked."""
    if not findings:
        return "  no routing pairs present; nothing to cross-check"
    lines = [f.line() for f in findings]
    failed = sum(1 for f in findings if f.verdict is Verdict.FAILED)
    unverified = sum(1 for f in findings if f.verdict is Verdict.UNVERIFIABLE)
    lines.append(f"  {len(findings)} check(s): {failed} failed, {unverified} unverifiable")
    return "\n".join(lines)


def main() -> int:
    """CLI: cross-check a payload, for the recipe book and for triage."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, help="JSON object of extracted fields")
    parser.add_argument(
        "--directory",
        type=__import__("pathlib").Path,
        default=None,
        help="JSON file of {iban: bic} confirmed correspondences",
    )
    args = parser.parse_args()

    payload = json.loads(args.payload)
    directory: RoutingDirectory = NoDirectory()
    if args.directory:
        data = json.loads(args.directory.read_text(encoding="utf-8"))
        directory = ObservedDirectory(pairs={_iban_key(k): v for k, v in data.items()})

    findings = cross_check(payload, directory)
    print(f"directory: {directory.name}")
    print(summarise(findings))
    return 1 if any(f.blocking for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
