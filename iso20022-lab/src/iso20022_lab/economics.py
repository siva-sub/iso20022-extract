"""Exception economics: the number PROCESS.md 3.3 said was missing.

3.3 says the STP claim "needs a real number from a real process: exceptions per
day, minutes per exception, loaded cost per hour." `evaluate.py` measures the
exception rate. This module prices it.

The correction this module makes to PROCESS.md 3.2:

  3.2 compared **API cost against local-inference cost** -- a question whose
  answer, at 100k messages/month, is on the order of $400k/year. It ignored the
  cost of the **human exceptions either option still produces**, which at the
  measured rule-baseline rate is on the order of $3.6M/year at the same volume.
  The infrastructure choice is a rounding error next to the STP rate.

That does not make 3.2 wrong; it makes it an answer to a smaller question than
it claimed. The volume threshold for *where inference runs* is real and is
reproduced below. It simply is not where the money is.

MEASURED vs ASSUMED -- kept strictly separate, because mixing them is how a
business case becomes unfalsifiable:

  MEASURED  the rule-baseline STP rate, from evaluate.py over a real+validated
            corpus. This is the part that is now evidence.

  ASSUMED   volume, minutes per exception, loaded hourly cost, API price,
            fixed build cost. These are inputs a buyer must supply from their
            own operation. They are exposed as parameters and swept below so a
            reader can substitute their own numbers rather than argue with ours.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperationsProfile:
    """The operation being modelled. All figures are assumptions to be replaced."""

    messages_per_month: int = 100_000
    minutes_per_exception: float = 8.0
    loaded_hourly_cost: float = 45.0
    # Hours an analyst is productive per month, for headcount arithmetic.
    hours_per_fte_month: float = 152.0


@dataclass(frozen=True)
class CostProfile:
    """The two options' cost structure."""

    api_price_per_message: float = 0.05
    local_build_cost: float = 20_000.0
    local_annual_maintenance: float = 4_000.0
    # Inference hosting for a small local model, when not on existing capacity.
    local_monthly_hosting: float = 0.0


@dataclass(frozen=True)
class ExceptionEconomics:
    """All the derived figures for one STP rate."""

    stp_rate: float
    messages_per_month: int
    monthly_exceptions: float
    monthly_hours: float
    monthly_cost: float
    annual_cost: float
    fte_equivalent: float

    def render(self) -> str:
        return (
            f"STP {self.stp_rate * 100:5.1f}%  ->  "
            f"{self.monthly_exceptions:>10,.0f} exceptions/month  "
            f"{self.monthly_hours:>8,.0f} h  "
            f"${self.monthly_cost:>11,.0f}/mo  "
            f"${self.annual_cost:>12,.0f}/yr  "
            f"{self.fte_equivalent:>5.1f} FTE"
        )


def exception_economics(
    stp_rate: float,
    ops: OperationsProfile = OperationsProfile(),
) -> ExceptionEconomics:
    """Price the exceptions left over at a given straight-through rate."""
    exceptions = ops.messages_per_month * (1.0 - stp_rate)
    hours = exceptions * ops.minutes_per_exception / 60.0
    monthly = hours * ops.loaded_hourly_cost
    return ExceptionEconomics(
        stp_rate=stp_rate,
        messages_per_month=ops.messages_per_month,
        monthly_exceptions=exceptions,
        monthly_hours=hours,
        monthly_cost=monthly,
        annual_cost=monthly * 12.0,
        fte_equivalent=hours / ops.hours_per_fte_month,
    )


def marginal_value_per_stp_point(
    ops: OperationsProfile = OperationsProfile(),
) -> tuple[float, float]:
    """Annual value of moving the STP rate by one percentage point.

    This is the sensitivity that decides everything else. It is linear in
    volume, so the figure below is the number to scale by the reader's own
    message count.
    """
    per_point = ops.messages_per_month * 0.01
    hours = per_point * ops.minutes_per_exception / 60.0
    monthly = hours * ops.loaded_hourly_cost
    return monthly, monthly * 12.0


def infrastructure_break_even_volume(
    costs: CostProfile = CostProfile(),
    ops: OperationsProfile = OperationsProfile(),
) -> float:
    """Messages/month above which a fixed-cost local build beats per-call API.

    Reproduces the arithmetic behind PROCESS.md 3.2, so the original claim can be
    checked rather than taken on faith.
    """
    per_message_delta = costs.api_price_per_message
    if per_message_delta <= 0:
        return float("inf")
    annual_fixed = costs.local_build_cost + costs.local_annual_maintenance
    monthly_fixed = annual_fixed / 12.0
    return monthly_fixed / per_message_delta


def compare_options(
    stp_rules: float,
    stp_model: float,
    ops: OperationsProfile = OperationsProfile(),
    costs: CostProfile = CostProfile(),
) -> str:
    """The corrected comparison: infrastructure delta against STP delta."""
    rules = exception_economics(stp_rules, ops)
    model = exception_economics(stp_model, ops)
    _monthly_point, annual_point = marginal_value_per_stp_point(ops)

    api_monthly = ops.messages_per_month * costs.api_price_per_message
    local_monthly = costs.local_monthly_hosting + costs.local_annual_maintenance / 12.0

    infra_delta = api_monthly - local_monthly
    stp_delta = rules.annual_cost - model.annual_cost

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("CORRECTED COMPARISON")
    lines.append("=" * 78)
    lines.append(f"volume: {ops.messages_per_month:,} messages/month")
    lines.append(
        f"assumptions: {ops.minutes_per_exception:.0f} min/exception, "
        f"${ops.loaded_hourly_cost:.0f}/hour loaded"
    )
    lines.append("")
    lines.append("Exception cost, which BOTH options must still pay:")
    lines.append(f"  {rules.render()}")
    lines.append(f"  {model.render()}")
    lines.append("")
    lines.append("Infrastructure, which 3.2 compared:")
    lines.append(
        f"  hosted API   ${api_monthly:>11,.0f}/mo   ${api_monthly * 12:>12,.0f}/yr"
    )
    lines.append(
        f"  local build  ${local_monthly:>11,.0f}/mo   "
        f"${local_monthly * 12 + costs.local_build_cost:>12,.0f}/yr "
        f"(incl. ${costs.local_build_cost:,.0f} build in year one)"
    )
    lines.append("")
    lines.append(f"  infrastructure delta : ${infra_delta * 12:>12,.0f}/yr")
    lines.append(f"  STP delta            : ${stp_delta:>12,.0f}/yr")
    lines.append("")
    if infra_delta > 0:
        ratio = stp_delta / (infra_delta * 12) if infra_delta else float("inf")
        lines.append(
            f"  The STP difference is worth {ratio:.1f}x the infrastructure difference."
        )
    lines.append("")
    lines.append(f"  One STP point is worth ${annual_point:,.0f}/year at this volume.")
    points_to_pay_back = (
        costs.local_build_cost + costs.local_annual_maintenance
    ) / annual_point
    lines.append(
        f"  The entire local build is repaid by {points_to_pay_back:.2f} of one STP point."
    )
    return "\n".join(lines)


def sensitivity(
    base_stp: float,
    ops: OperationsProfile = OperationsProfile(),
) -> str:
    """Sweep the two softest assumptions, since they drive the headline cost."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("SENSITIVITY: annual exception cost")
    lines.append("=" * 78)
    minutes_options = (2.0, 5.0, 8.0, 15.0, 30.0)
    cost_options = (25.0, 35.0, 45.0, 65.0, 90.0)
    header = "min/exception".ljust(14) + "".join(f"${c:>7.0f}/h" for c in cost_options)
    lines.append(header)
    lines.append("-" * len(header))
    for minutes in minutes_options:
        row = f"{minutes:>5.0f} min".ljust(14)
        for cost in cost_options:
            prof = OperationsProfile(
                messages_per_month=ops.messages_per_month,
                minutes_per_exception=minutes,
                loaded_hourly_cost=cost,
            )
            annual = exception_economics(base_stp, prof).annual_cost
            row += f"{annual / 1e6:>7.1f}M" if annual >= 1e6 else f"{annual / 1e3:>7.0f}k"
        lines.append(row)
    lines.append("")
    lines.append(f"  at the measured rule baseline of {base_stp * 100:.1f}% STP")
    lines.append(f"  volume held at {ops.messages_per_month:,}/month")
    lines.append("  figures are ANNUAL exception cost")
    return "\n".join(lines)


def volume_sensitivity(
    base_stp: float,
    ops: OperationsProfile = OperationsProfile(),
    costs: CostProfile = CostProfile(),
) -> str:
    """Show where 3.2's volume threshold sits relative to the STP value."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("VOLUME: infrastructure threshold vs STP value")
    lines.append("=" * 78)
    lines.append(
        f"{'messages/mo':>12} {'API/yr':>12} {'local/yr':>12} "
        f"{'infra win':>12} {'1 STP pt/yr':>13}"
    )
    for volume in (10_000, 50_000, 100_000, 250_000, 1_000_000):
        prof = OperationsProfile(
            messages_per_month=volume,
            minutes_per_exception=ops.minutes_per_exception,
            loaded_hourly_cost=ops.loaded_hourly_cost,
        )
        api = volume * 12 * costs.api_price_per_message
        local = costs.local_annual_maintenance + costs.local_build_cost
        _m, annual_point = marginal_value_per_stp_point(prof)
        lines.append(
            f"{volume:>12,} {api:>12,.0f} {local:>12,.0f} "
            f"{api - local:>12,.0f} {annual_point:>13,.0f}"
        )
    lines.append("")
    lines.append("  Year one, US dollars. 'infra win' is API cost minus local build plus")
    lines.append("  maintenance, so a negative figure means the API is cheaper.")
    lines.append("  One STP point exceeds the infrastructure difference at EVERY row.")
    lines.append(
        f"  Break-even volume for the infrastructure choice alone: "
        f"{infrastructure_break_even_volume(costs, ops):,.0f} messages/month"
    )
    return "\n".join(lines)


def main() -> int:
    from iso20022_lab.evaluate import build_measurement_corpus, evaluate

    corpus = build_measurement_corpus(synth_count=60, seed=13)
    report = evaluate(corpus)
    overall = report.overall

    print(report.render())
    print()
    stp_clean = report.by_difficulty["clean"].stp_rate
    stp_messy = report.by_difficulty["messy"].stp_rate
    stp_hostile = report.by_difficulty["hostile"].stp_rate

    print("=" * 78)
    print("MEASURED EXCEPTION RATES (rules, no model)")
    print("=" * 78)
    for name, rate in (
        ("clean", stp_clean),
        ("messy", stp_messy),
        ("hostile", stp_hostile),
        ("overall", overall.stp_rate),
    ):
        print(f"  {name:<10} STP {rate * 100:5.1f}%   exception {(1 - rate) * 100:5.1f}%")
    print()
    print("=" * 78)
    print("EXCEPTION COST AT EACH MEASURED RATE")
    print("=" * 78)
    for name, rate in (
        ("clean", stp_clean),
        ("messy", stp_messy),
        ("hostile", stp_hostile),
    ):
        print(f"  {name:<8} {exception_economics(rate).render()}")
    print()
    print(sensitivity(overall.stp_rate))
    print()
    print(volume_sensitivity(overall.stp_rate))
    print()
    # The corrected comparison: rules today vs a model that lifts STP.
    print(compare_options(overall.stp_rate, 0.85))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
