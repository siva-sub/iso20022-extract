"""Why this cannot be a regex parser: the economics of format concentration.

The measured workflow result is a **step function, not a gradient**:

    rules_v1 (written for ONE format):   100% STP on that format, 0% on all others
    rules_v2 (every synonym, best effort): 92.5% clean, 62.5% messy, 0% hostile

Regex does not "fail gradually". It succeeds completely inside the format it was
written for and fails completely outside it. So the cost of a rules approach is
not a percentage -- it is **one rule set per document format**, and the number of
formats is a property of your customer base, not your volume.

That is the whole economic argument, and it is why the question "why not regex"
has a real answer rather than a sales answer:

  * Rules cost scales with **format count** (engineering, per customer).
  * Model cost is **fixed** (one artifact, one integration).
  * Human cost is what both are trying to avoid, and it is concentrated in the
    long tail -- the formats nobody has written rules for and no analyst has
    memorised.

For a shop with three formats, regex wins outright and buying a model is waste.
For a shop with two hundred, the rule maintenance alone exceeds the model. This
module computes where that crossover sits so the claim can be checked rather than
asserted.

This module is the code behind PROCESS.md section 17. That section is the
readable version and carries the measured table, the sensitivity sweep, the
limitations, and the decision procedure; read it before acting on a number here.

Second argument, and the one that is usually missed: **a new customer onboarding**.

  * Rules: an engineer writes, tests and deploys a rule set. Days, per customer.
  * Model: zero. The next format is just another document.

That is time-to-revenue, not headcount, and on a per-customer basis it dominates
everything else in this file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleCost:
    """What one additional document format costs the rules approach."""

    engineer_days_per_format: float = 2.0
    maintenance_days_per_format_year: float = 0.5
    loaded_engineer_day: float = 600.0

    def year_one(self, formats: int) -> float:
        return formats * self.engineer_days_per_format * self.loaded_engineer_day

    def annual_run(self, formats: int) -> float:
        return formats * self.maintenance_days_per_format_year * self.loaded_engineer_day


@dataclass(frozen=True)
class ModelCost:
    """What the model approach costs. One artifact, no per-format term."""

    build_days: float = 25.0
    loaded_engineer_day: float = 600.0
    annual_maintenance_days: float = 6.0
    # Integration is real work and is charged once, not per format.
    integration_days: float = 10.0

    def year_one(self, formats: int = 1) -> float:
        _ = formats
        return (self.build_days + self.integration_days) * self.loaded_engineer_day

    def annual_run(self, formats: int = 1) -> float:
        _ = formats
        return self.annual_maintenance_days * self.loaded_engineer_day


@dataclass(frozen=True)
class HumanCost:
    """Analyst effort, which depends on how well the tail is covered."""

    minutes_key: float = 8.0
    minutes_verify: float = 1.5
    loaded_hourly: float = 45.0
    messages_per_month: int = 100_000

    def annual(
        self,
        stp_rate: float,
        verify_rate: float,
    ) -> float:
        """Annual analyst cost for a given STP and verify mix.

        `stp_rate` needs no human. Of the remainder, `verify_rate` is corrected
        rather than keyed from scratch.
        """
        exceptions = self.messages_per_month * 12 * (1.0 - stp_rate)
        keyed = exceptions * (1.0 - verify_rate)
        verified = exceptions * verify_rate
        minutes = keyed * self.minutes_key + verified * self.minutes_verify
        return minutes / 60.0 * self.loaded_hourly


def coverage(
    formats: int,
    concentration: float,
    rules_cover_top: int,
) -> tuple[float, float]:
    """Fraction of VOLUME that rules cover, given how concentrated formats are.

    Volume is assumed Zipf-like over formats, which is how inbound actually
    behaves: a handful of templates dominate and a long tail trickles.

    `concentration` is the Zipf exponent. Higher means a few formats carry most
    of the volume, which is the case that makes rules look good.

    Returns (rule_covered_share, tail_share).
    """
    weights = [1.0 / (i + 1) ** concentration for i in range(formats)]
    total = sum(weights)
    covered = sum(weights[:rules_cover_top])
    return covered / total, (total - covered) / total


def compare(
    formats: int,
    concentration: float = 1.0,
    rules_cover_top: int = 3,
    rule: RuleCost = RuleCost(),
    model: ModelCost = ModelCost(),
    human: HumanCost = HumanCost(),
    model_stp: float = 0.85,
) -> dict[str, float]:
    """Year-one and annual cost of both approaches at a given format count.

    STP assumptions are stated, not smuggled. Rules achieve 100% STP inside the
    formats they cover and 0% outside, which is the measured step function.
    """
    covered, tail = coverage(formats, concentration, rules_cover_top)

    # Within a covered format rules are exact; outside them they produce nothing
    # usable, which is a KEY outcome rather than a VERIFY one.
    rules_stp = covered
    rules_verify = 0.0

    rules_human_y1 = human.annual(rules_stp, rules_verify)
    model_human_y1 = human.annual(model_stp, 1.0 - model_stp)

    return {
        "formats": float(formats),
        "rule_volume_covered": covered,
        "tail_share": tail,
        "rules_year_one": rule.year_one(formats) + rules_human_y1,
        "model_year_one": model.year_one(formats) + model_human_y1,
        "rules_annual_run": rule.annual_run(formats) + rules_human_y1,
        "model_annual_run": model.annual_run(formats) + model_human_y1,
        "rules_stp": rules_stp,
        "model_stp": model_stp,
        "rules_human_annual": rules_human_y1,
        "model_human_annual": model_human_y1,
    }


def crossover(
    concentration: float = 1.0,
    rules_cover_top: int = 3,
    rule: RuleCost = RuleCost(),
    model: ModelCost = ModelCost(),
    human: HumanCost = HumanCost(),
    model_stp: float = 0.85,
    upper: int = 500,
) -> int:
    """Smallest format count at which the model is cheaper in year one.

    Scanned rather than solved because the human term depends on a Zipf sum, so
    there is no closed form worth the algebra.
    """
    for formats in range(1, upper + 1):
        row = compare(
            formats, concentration, rules_cover_top, rule, model, human, model_stp
        )
        if row["model_year_one"] < row["rules_year_one"]:
            return formats
    return upper + 1


def onboarding_table(rule: RuleCost, model: ModelCost) -> str:
    """Cost and lead time of adding ONE more customer with a new template."""
    lines: list[str] = []
    lines.append("Adding one more customer, one more document format:")
    lines.append("")
    lines.append(f"  {'':<22} {'rules':>14} {'model':>14}")
    lines.append(
        f"  {'engineering days':<22} {rule.engineer_days_per_format:>14.1f} {0.0:>14.1f}"
    )
    lines.append(
        f"  {'cost':<22} "
        f"${rule.engineer_days_per_format * rule.loaded_engineer_day:>13,.0f} "
        f"${0.0:>13,.0f}"
    )
    lines.append(f"  {'lead time':<22} {'days to deploy':>14} {'none':>14}")
    lines.append("")
    lines.append("  The model's marginal cost per new format is zero, and so is its lead")
    lines.append("  time. That is time-to-revenue on every new customer, and it is the")
    lines.append("  argument that survives scrutiny even when accuracy is close.")
    return "\n".join(lines)


def render(
    rule: RuleCost = RuleCost(),
    model: ModelCost = ModelCost(),
    human: HumanCost = HumanCost(),
) -> str:
    lines: list[str] = []
    lines.append("=" * 86)
    lines.append("ECONOMICS: cost scales with FORMAT COUNT, not with message volume")
    lines.append("=" * 86)
    lines.append(
        f"volume {human.messages_per_month:,}/month | "
        f"KEY {human.minutes_key:.0f} min | VERIFY {human.minutes_verify:.1f} min | "
        f"${human.loaded_hourly:.0f}/hour"
    )
    lines.append("")
    lines.append(
        f"{'formats':>8} {'vol covered':>12} {'rules Y1':>14} {'model Y1':>14} "
        f"{'cheaper':>10}"
    )
    for formats in (1, 3, 5, 10, 25, 50, 100, 250):
        for conc, label in ((1.0, "zipf-1"),):
            row = compare(formats, conc, 3, rule, model, human)
            r = row["rules_year_one"]
            m = row["model_year_one"]
            cheaper = "rules" if r < m else "model"
            lines.append(
                f"{formats:>8} {row['rule_volume_covered'] * 100:>11.1f}% "
                f"${r:>13,.0f} ${m:>13,.0f} {cheaper:>10}"
            )
            _ = label
    lines.append("")
    lines.append("crossover (formats at which the model becomes cheaper in year one):")
    for conc in (0.6, 1.0, 1.5):
        n = crossover(conc, 3, rule, model, human)
        lines.append(f"  concentration {conc:>4.1f} -> model cheaper from {n:>4} formats")
    lines.append("")
    lines.append("  Lower concentration means volume is spread across more formats,")
    lines.append("  so the tail rules cannot reach is larger and the model wins sooner.")
    return "\n".join(lines)


def main() -> int:
    print(render())
    print()
    print(onboarding_table(RuleCost(), ModelCost()))
    print()
    print("=" * 86)
    print("SENSITIVITY: does the answer survive a pessimistic model assumption?")
    print("=" * 86)
    print(f"  {'model STP':>10} {'crossover formats':>20}")
    for stp in (0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
        n = crossover(1.0, 3, RuleCost(), ModelCost(), HumanCost(), model_stp=stp)
        print(f"  {stp * 100:>9.0f}% {n:>20}")
    print()
    print("  Even at 60% STP, well below what the rule baseline achieves on messy")
    print("  documents, the model is cheaper once the format count is large. The")
    print("  argument does not depend on the model being more accurate than rules")
    print("  inside a format -- it depends on rules needing one ruleset per format.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
