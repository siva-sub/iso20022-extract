"""Tests for `routing.py`: cross-checking unchecksummed fields.

Every case in `THE REAL FAILURES` below is a value taken from the measured
PP-OCRv6 run in PROCESS.md 18.12 -- not invented for the test. That matters,
because the whole justification for this module is an observed failure mode, and
a synthetic example would let the check drift away from the thing it was built
for while the suite stayed green.

The load-bearing assertions are the ones about `UNVERIFIABLE`. A check that
reports "passed" when it had nothing to compare against is worse than no check,
because it converts an unknown into a false assurance. Several tests here exist
solely to pin that distinction down.
"""

from __future__ import annotations

from iso20022_lab.routing import (
    NoDirectory,
    ObservedDirectory,
    Verdict,
    bic_bank_code,
    bic_country,
    check_country_agreement,
    cross_check,
    iban_country,
    summarise,
)

# The measured silent failures. Truth on the left, what recognition produced on
# the right. Both were graded "silent" -- nothing in the pipeline rejected them.
FR_TO_ER = ("BNPAFRPP704", "BNPAERPP704", "FR7630006000011234567890189")
NL_TO_NI = ("ABNANL2A836", "ARNANI2A836", "NL9439110653356593")
# Country survived; only the institution code was damaged. No country check can
# see this one, which is why the directory exists.
DE_BANK_DAMAGED = ("DEUTDEFF349", "DEUTDEEE349", "DE89370400440532013000")


# --------------------------------------------------------------------------
# Extractors
# --------------------------------------------------------------------------


def test_iban_country_reads_the_prefix() -> None:
    assert iban_country("FR7630006000011234567890189") == "FR"
    assert iban_country("NL94 3911 0653 3565 93") == "NL", "spaces are tolerated"


def test_bic_country_is_positions_five_and_six() -> None:
    assert bic_country("DEUTDEFF") == "DE"
    assert bic_country("BNPAFRPP704") == "FR"


def test_bank_code_is_not_the_same_as_an_iban_bank_code() -> None:
    """`DEUT` and a German BLZ are related by a directory, not by a rule.

    Pinning the distinction keeps anyone from later "simplifying" a directory
    lookup into a string comparison that would be quietly wrong.
    """
    assert bic_bank_code("DEUTDEFF349") == "DEUT"
    # A German BLZ is eight digits and does not appear in the BIC at all.
    assert "37040044" not in "DEUTDEFF349"


def test_short_values_do_not_raise() -> None:
    assert bic_country("DE") == ""
    assert iban_country("") == ""


# --------------------------------------------------------------------------
# The real failures, now caught
# --------------------------------------------------------------------------


def test_france_read_as_eritrea_is_caught() -> None:
    """`FR` -> `ER` is a valid BIC for Eritrea, which is why shape checks missed it."""
    truth_bic, ocr_bic, iban = FR_TO_ER
    assert bic_country(truth_bic) == "FR"

    findings = cross_check({"CdtrAcct_IBAN": iban, "CdtrAgt_BICFI": ocr_bic})
    assert any(f.blocking for f in findings), summarise(findings)

    # And the correct value must NOT be flagged, or the check is useless.
    clean = cross_check({"CdtrAcct_IBAN": iban, "CdtrAgt_BICFI": truth_bic})
    assert not any(f.blocking for f in clean), summarise(clean)


def test_netherlands_read_as_nicaragua_is_caught() -> None:
    truth_bic, ocr_bic, iban = NL_TO_NI
    assert bic_country(ocr_bic) == "NI", "NI is Nicaragua, a real country"

    assert any(
        f.blocking for f in cross_check({"CdtrAcct_IBAN": iban, "CdtrAgt_BICFI": ocr_bic})
    )
    assert not any(
        f.blocking for f in cross_check({"CdtrAcct_IBAN": iban, "CdtrAgt_BICFI": truth_bic})
    )


def test_country_check_alone_cannot_catch_a_same_country_damage() -> None:
    """The honest limit, asserted so it cannot be mistaken for full coverage.

    `DEUTDEEE349` is a well-formed German BIC for an institution that does not
    exist. Country agreement passes, because both are `DE`. Only a directory can
    see this, and this test states that rather than leaving it implied.
    """
    _, ocr_bic, iban = DE_BANK_DAMAGED
    findings = cross_check({"CdtrAcct_IBAN": iban, "CdtrAgt_BICFI": ocr_bic})

    assert not any(f.blocking for f in findings), (
        "country agreement must pass here -- that is the gap being documented"
    )
    assert any(f.verdict is Verdict.UNVERIFIABLE for f in findings), (
        "the unchecked bank code must be reported as unverifiable, not as passed"
    )


def test_a_directory_catches_the_damage_the_country_check_misses() -> None:
    """The residual class, closed by the payment's own routing data."""
    truth_bic, ocr_bic, iban = DE_BANK_DAMAGED
    directory = ObservedDirectory()
    directory.observe(iban, truth_bic)  # confirmed at settlement

    bad = cross_check({"CdtrAcct_IBAN": iban, "CdtrAgt_BICFI": ocr_bic}, directory)
    assert any(f.blocking for f in bad), summarise(bad)
    assert "different institution" in summarise(bad)

    good = cross_check({"CdtrAcct_IBAN": iban, "CdtrAgt_BICFI": truth_bic}, directory)
    assert not any(f.blocking for f in good), summarise(good)


# --------------------------------------------------------------------------
# UNVERIFIABLE must never be reported as PASSED
# --------------------------------------------------------------------------


def test_no_directory_reports_unverifiable_not_passed() -> None:
    findings = cross_check(
        {"CdtrAcct_IBAN": "FR7630006000011234567890189", "CdtrAgt_BICFI": "BNPAFRPP704"}
    )
    directory_findings = [f for f in findings if f.check == "bic_matches_directory"]
    assert directory_findings, "the bank code check must run, and say it could not"
    assert all(f.verdict is Verdict.UNVERIFIABLE for f in directory_findings)
    assert not any(f.verdict is Verdict.PASSED for f in directory_findings)


def test_an_unknown_iban_is_unverifiable_even_with_a_directory() -> None:
    directory = ObservedDirectory()
    directory.observe("DE89370400440532013000", "DEUTDEFF")
    findings = cross_check(
        {"CdtrAcct_IBAN": "FR7630006000011234567890189", "CdtrAgt_BICFI": "BNPAFRPP704"},
        directory,
    )
    check = [f for f in findings if f.check == "bic_matches_directory"]
    assert check and check[0].verdict is Verdict.UNVERIFIABLE


def test_an_invalid_iban_cannot_be_used_as_the_reference() -> None:
    """A corrupted IBAN fails mod-97, so it cannot vouch for the BIC.

    Marking this UNVERIFIABLE rather than PASSED matters: the comparison would
    otherwise be against a country prefix that was itself misread.
    """
    finding = check_country_agreement(
        "BNPAFRPP704", "FR76OOO6000011234567890189", "creditor", NoDirectory()
    )
    assert finding.verdict is Verdict.UNVERIFIABLE
    assert "mod-97" in finding.detail


def test_a_malformed_bic_is_a_failure_not_a_mismatch() -> None:
    finding = check_country_agreement(
        "not-a-bic", "FR7630006000011234567890189", "creditor", NoDirectory()
    )
    assert finding.verdict is Verdict.FAILED
    assert finding.check == "bic_shape"


# --------------------------------------------------------------------------
# Pairing: the BIC must be compared against the RIGHT account
# --------------------------------------------------------------------------


def test_agent_bic_is_paired_with_its_own_party() -> None:
    """A cross-party mix-up must not pass by being compared to the other IBAN.

    This is the wrong-recipient failure, and it is the one error nothing
    downstream catches -- so the check must not be able to cancel itself out by
    picking whichever IBAN happens to agree.
    """
    payload = {
        # Creditor's bank is French and its account is French: consistent.
        "CdtrAcct_IBAN": "FR7630006000011234567890189",
        "CdtrAgt_BICFI": "BNPAFRPP704",
        # Debtor's BIC says Germany but the debtor's account is Dutch: not.
        "DbtrAcct_IBAN": "NL9439110653356593",
        "DbtrAgt_BICFI": "DEUTDEFF349",
    }
    findings = cross_check(payload)
    failed = [f for f in findings if f.blocking]
    assert len(failed) == 1
    assert failed[0].party == "debtor"
    assert "DE" in failed[0].detail and "NL" in failed[0].detail


def test_missing_fields_are_skipped_not_failed() -> None:
    """A field the recogniser never produced is a completeness problem.

    Blaming the routing data for an absent field would report a routing
    disagreement that did not happen.
    """
    assert cross_check({}) == []
    assert cross_check({"CdtrAcct_IBAN": "FR7630006000011234567890189"}) == []
    assert cross_check({"CdtrAgt_BICFI": "BNPAFRPP704"}) == []


def test_each_pair_is_checked_once() -> None:
    payload = {
        "CdtrAcct_IBAN": "FR7630006000011234567890189",
        "CdtrAgt_BICFI": "BNPAFRPP704",
        "DbtrAcct_IBAN": "NL9439110653356593",
        "DbtrAgt_BICFI": "ABNANL2A836",
    }
    findings = cross_check(payload)
    countries = [f for f in findings if f.check == "bic_country_matches_iban"]
    assert len(countries) == 2, "one per party, no duplicates"


# --------------------------------------------------------------------------
# The directory
# --------------------------------------------------------------------------


def test_observed_directory_normalises_keys() -> None:
    d = ObservedDirectory()
    d.observe("DE89 3704 0044 0532 0130 00", "deutdeff")
    assert d.bic_for_iban("DE89370400440532013000") == "DEUTDEFF"


def test_observed_directory_does_not_learn_from_predictions() -> None:
    """It has an explicit `observe` rather than a learn-from-output path.

    A directory built from a model's own predictions would rediscover the model's
    errors as routing rules, and the check would then confirm them. This asserts
    the positive API and that mere construction learns nothing.
    """
    d = ObservedDirectory()
    assert len(d) == 0
    d.observe("DE89370400440532013000", "DEUTDEFF349")
    assert len(d) == 1


def test_empty_directory_is_not_the_same_as_no_directory() -> None:
    """Both say unverifiable, but they are different deployment states."""
    payload = {
        "CdtrAcct_IBAN": "FR7630006000011234567890189",
        "CdtrAgt_BICFI": "BNPAFRPP704",
    }
    none = summarise(cross_check(payload, NoDirectory()))
    empty = summarise(cross_check(payload, ObservedDirectory()))
    assert "none" in none and "observed" in empty
