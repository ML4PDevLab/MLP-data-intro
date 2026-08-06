#!/usr/bin/env python3
"""Generate the public-facing MLP data coverage and indicator guide.

The script intentionally uses only the Python standard library so maintainers can
refresh the documentation without installing the analysis environment.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FINAL = DATA / "final-counts"
OUTPUT = ROOT / "docs" / "DATA_COVERAGE_AND_INDICATORS.md"
COUNTRY_CSV = ROOT / "docs" / "data_coverage" / "country_coverage.csv"
SOURCE_CSV = ROOT / "docs" / "data_coverage" / "source_coverage.csv"

DATASETS = {
    "Civic Space": FINAL / "full-civic-data.csv",
    "RAI": FINAL / "full-rai-data.csv",
    "MLEED": FINAL / "full-mleed-data.csv",
}

# ISO 3166-1 alpha-3 codes, except XKX, which is the commonly used provisional
# code for Kosovo. Repository country names are preserved exactly as stored.
ISO3 = {
    "Albania": "ALB",
    "Algeria": "DZA",
    "Angola": "AGO",
    "Armenia": "ARM",
    "Azerbaijan": "AZE",
    "Bangladesh": "BGD",
    "Belarus": "BLR",
    "Benin": "BEN",
    "Bolivia": "BOL",
    "Brazil": "BRA",
    "Burkina Faso": "BFA",
    "Cambodia": "KHM",
    "Cameroon": "CMR",
    "Colombia": "COL",
    "Costa Rica": "CRI",
    "DR Congo": "COD",
    "Dominican Republic": "DOM",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "El Salvador": "SLV",
    "Ethiopia": "ETH",
    "Georgia": "GEO",
    "Ghana": "GHA",
    "Guatemala": "GTM",
    "Honduras": "HND",
    "Hungary": "HUN",
    "India": "IND",
    "Indonesia": "IDN",
    "Jamaica": "JAM",
    "Kazakhstan": "KAZ",
    "Kenya": "KEN",
    "Kosovo": "XKX",
    "Kyrgyzstan": "KGZ",
    "Liberia": "LBR",
    "Macedonia": "MKD",
    "Malawi": "MWI",
    "Malaysia": "MYS",
    "Mali": "MLI",
    "Mauritania": "MRT",
    "Mexico": "MEX",
    "Moldova": "MDA",
    "Morocco": "MAR",
    "Mozambique": "MOZ",
    "Namibia": "NAM",
    "Nepal": "NPL",
    "Nicaragua": "NIC",
    "Niger": "NER",
    "Nigeria": "NGA",
    "Pakistan": "PAK",
    "Panama": "PAN",
    "Paraguay": "PRY",
    "Peru": "PER",
    "Philippines": "PHL",
    "Rwanda": "RWA",
    "Senegal": "SEN",
    "Serbia": "SRB",
    "Solomon Islands": "SLB",
    "South Africa": "ZAF",
    "South Sudan": "SSD",
    "Sri Lanka": "LKA",
    "Tanzania": "TZA",
    "Timor Leste": "TLS",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "Uganda": "UGA",
    "Ukraine": "UKR",
    "Uzbekistan": "UZB",
    "Zambia": "ZMB",
    "Zimbabwe": "ZWE",
}

FINAL_METADATA_COLUMNS = {
    "country",
    "influencer",
    "date",
    "year",
    "month",
    "total_articles",
    "total_from_source",
    "total_label_events",
    "total_local_docs",
}


@dataclass
class CountryPeriod:
    dates: set[str] = field(default_factory=set)

    @property
    def start(self) -> str:
        return min(self.dates) if self.dates else ""

    @property
    def end(self) -> str:
        return max(self.dates) if self.dates else ""


@dataclass
class DatasetSummary:
    name: str
    path: Path
    records: int
    fields: list[str]
    countries: dict[str, CountryPeriod]

    @property
    def start(self) -> str:
        return min(period.start for period in self.countries.values() if period.start)

    @property
    def end(self) -> str:
        return max(period.end for period in self.countries.values() if period.end)


@dataclass
class SourcePeriod:
    country: str
    source: str
    dates: set[str] = field(default_factory=set)
    total_articles: float = 0

    @property
    def start(self) -> str:
        return min(self.dates) if self.dates else ""

    @property
    def end(self) -> str:
        return max(self.dates) if self.dates else ""


def read_dataset(name: str, path: Path) -> DatasetSummary:
    countries: dict[str, CountryPeriod] = defaultdict(CountryPeriod)
    records = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for row in reader:
            country = (row.get("country") or "").strip()
            month = (row.get("date") or "").strip()
            if country and month:
                countries[country].dates.add(month)
            records += 1
    return DatasetSummary(name, path, records, fields, dict(countries))


def read_indicator_dictionary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def mleed_indicators(fields: list[str]) -> tuple[list[str], list[str]]:
    raw = [
        field_name
        for field_name in fields
        if field_name not in FINAL_METADATA_COLUMNS
        and "Norm" not in field_name
    ]
    named = [field_name for field_name in raw if "999" not in field_name]
    unclassified = [field_name for field_name in raw if "999" in field_name]
    return named, unclassified


def read_r_character_vector(path: Path, variable: str) -> set[str]:
    """Read a simple quoted c(...) vector from an R source file."""
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"\b{re.escape(variable)}\s*(?:<-|=)\s*c\((.*?)\)",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"Could not find {variable} in {path}")
    return {
        value.removesuffix(".csv")
        for value in re.findall(r'[\"\']([^\"\']+)[\"\']', match.group(1))
    }


def source_type(source: str, international: set[str], regional: set[str]) -> str:
    if source in international:
        return "International"
    if source in regional:
        return "Regional"
    return "Local"


def read_source_coverage() -> dict[tuple[str, str], SourcePeriod]:
    coverage: dict[tuple[str, str], SourcePeriod] = {}
    source_dir = DATA / "0-civic-by-source"
    for path in sorted(source_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                country = (row.get("country") or path.stem).strip()
                source = (row.get("source") or "").strip()
                if not country or not source:
                    continue
                key = (country, source)
                period = coverage.setdefault(key, SourcePeriod(country, source))
                try:
                    articles = float(row.get("total_articles") or 0)
                except ValueError:
                    articles = 0
                if articles > 0:
                    month = (row.get("date") or "").strip()
                    if month:
                        period.dates.add(month)
                    period.total_articles += articles
    return coverage


def month_label(value: str) -> str:
    return value[:7] if value else "—"


def period_label(period: CountryPeriod | SourcePeriod | None) -> str:
    if period is None or not period.dates:
        return "—"
    return f"{month_label(period.start)}–{month_label(period.end)} ({len(period.dates)} months)"


def integer_label(value: float) -> str:
    return f"{int(value):,}" if value.is_integer() else f"{value:,.2f}"


def md_escape(value: str) -> str:
    return value.replace("|", "\\|")


def write_country_csv(summaries: dict[str, DatasetSummary], countries: list[str]) -> None:
    COUNTRY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with COUNTRY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "country",
                "iso3",
                "civic_start",
                "civic_end",
                "civic_months",
                "rai_start",
                "rai_end",
                "rai_months",
                "mleed_start",
                "mleed_end",
                "mleed_months",
            ]
        )
        for country in countries:
            row = [country, ISO3.get(country, "")]
            for dataset_name in ("Civic Space", "RAI", "MLEED"):
                period = summaries[dataset_name].countries.get(country, CountryPeriod())
                row.extend([period.start, period.end, len(period.dates)])
            writer.writerow(row)


def write_source_csv(
    coverage: dict[tuple[str, str], SourcePeriod], source_types: dict[str, str]
) -> None:
    SOURCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SOURCE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "country",
                "source",
                "source_type",
                "active_start",
                "active_end",
                "active_months",
                "total_articles",
            ]
        )
        for key in sorted(coverage):
            period = coverage[key]
            writer.writerow(
                [
                    period.country,
                    period.source,
                    source_types[period.source],
                    period.start,
                    period.end,
                    len(period.dates),
                    integer_label(period.total_articles).replace(",", ""),
                ]
            )


def build_markdown(
    summaries: dict[str, DatasetSummary],
    countries: list[str],
    civic: list[dict[str, str]],
    rai: list[dict[str, str]],
    mleed: list[str],
    mleed_unclassified: list[str],
    coverage: dict[tuple[str, str], SourcePeriod],
    source_types: dict[str, str],
    international_count: int,
) -> str:
    lines: list[str] = []
    add = lines.append
    generated = date.today().isoformat()

    add("# MLP Data Coverage and Indicator Guide")
    add("")
    add("> **Draft — maintainer review required before publication.** This guide is generated from the current repository exports, not from a live database. Resolve the review flags below before linking it from the public data-request form.")
    add("")
    add(f"Data snapshot inspected: **{generated}**")
    add("")
    add("This guide helps prospective users determine whether the Machine Learning for Peace (MLP) and Machine Learning for Environmental Event Detection (MLEED) data are likely to meet their needs before submitting a request. It documents observed country and date coverage, indicator names, and the news-source inventory contained in the current repository exports.")
    add("")
    add("## Public project resources")
    add("")
    add("- [Civic Space dashboard](https://web.sas.upenn.edu/mlp-devlab/civic-space-data-and-forecasts/overview/)")
    add("- [Resurgent Authoritarian Influence (RAI) dashboard](https://web.sas.upenn.edu/mlp-devlab/rai/rai-overview/)")
    add("- [Machine Learning for Environmental Event Detection (MLEED) dashboard](https://web.sas.upenn.edu/mlp-devlab/environmental-event-and-tracking/overview/)")
    add("- Questions: [zungru@sas.upenn.edu](mailto:zungru@sas.upenn.edu)")
    add("")
    add("## Maintainer review flags")
    add("")
    review_flags = [
        f"The current union of combined exports contains **{len(countries)} countries**. Existing project text elsewhere mentions 65 or 71 countries; public wording should identify whether it refers to all project datasets or a particular dataset release.",
        f"`data/rai_vars.csv` defines **{len(rai)} named RAI indicators**. The output also contains an unclassified `-999` field, while older documentation describes 22 categories.",
        f"The current MLEED output contains **{len(mleed)} named fields** and **{len(mleed_unclassified)} unclassified fields**. This should be reconciled with public wording that describes 16 event types.",
    ]
    source_countries = {period.country for period in coverage.values()}
    missing_source_countries = sorted(set(countries) - source_countries)
    if missing_source_countries:
        missing_labels = ", ".join(
            f"{country} ({ISO3.get(country, 'unmapped')})"
            for country in missing_source_countries
        )
        review_flags.append(
            f"Civic/RAI source-level extracts are absent for: **{missing_labels}**. The named coverage still appears in at least one combined output."
        )
    future_outputs = [
        f"{summary.name} ({month_label(summary.end)})"
        for summary in summaries.values()
        if month_label(summary.end) > generated[:7]
    ]
    if future_outputs:
        review_flags.append(
            f"The current export includes dates after the guide-generation month: **{', '.join(future_outputs)}**. Confirm whether these rows are forecasts, partial periods, or dated event data."
        )
    for number, flag in enumerate(review_flags, start=1):
        add(f"{number}. {flag}")
    add("")
    add("## Data structure and interpretation")
    add("")
    add("- The principal temporal unit is the **country-month**. RAI adds an `influencer` dimension (`china`, `russia`, and `combined`).")
    add("- Event measures are based on classified news articles. They capture the volume or relative prominence of reporting in the corpus; they should not automatically be interpreted as counts of unique real-world events.")
    add("- Columns ending in `Norm` are normalized measures. Columns ending in `NormShock` are shock indicators generated by the project pipeline.")
    add("- Fields containing `-999` are unclassified/other model outputs and are not counted as named indicators in this guide.")
    add("- Coverage dates below are observed in the current files. They do not guarantee that every source is active in every intervening month.")
    add("")
    add("## Dataset snapshot")
    add("")
    add("| Dataset | File | Countries | Observed range | Records | Columns |")
    add("|---|---|---:|---|---:|---:|")
    for summary in summaries.values():
        relative = summary.path.relative_to(ROOT).as_posix()
        add(f"| {summary.name} | [`{relative}`](../{relative}) | {len(summary.countries)} | {month_label(summary.start)}–{month_label(summary.end)} | {summary.records:,} | {len(summary.fields)} |")
    add("")
    partial_countries = []
    for country in countries:
        present = [
            dataset_name
            for dataset_name, summary in summaries.items()
            if country in summary.countries
        ]
        if len(present) != len(summaries):
            partial_countries.append(
                f"- **{country} (`{ISO3.get(country, 'unmapped')}`):** {', '.join(present)} only"
            )
    if partial_countries:
        add("### Dataset-specific country additions")
        add("")
        add("The following countries are included in the project-wide union but not in all three combined datasets:")
        add("")
        lines.extend(partial_countries)
        add("")
    add("A machine-readable version of the next table is available as [`country_coverage.csv`](data_coverage/country_coverage.csv).")
    add("")
    add("## Country and monthly coverage")
    add("")
    add("| Country | ISO3 | Civic Space | RAI | MLEED |")
    add("|---|---|---|---|---|")
    for country in countries:
        add(
            f"| {md_escape(country)} | `{ISO3.get(country, '—')}` | {period_label(summaries['Civic Space'].countries.get(country))} "
            f"| {period_label(summaries['RAI'].countries.get(country))} "
            f"| {period_label(summaries['MLEED'].countries.get(country))} |"
        )
    add("")
    add("## Civic Space indicators")
    add("")
    add(f"The current indicator dictionary contains **{len(civic)} named indicators**. The final combined Civic Space file provides normalized columns; the source-level files retain raw article counts. Some pipeline outputs also include `cr_` and `nr_` relevance variants.")
    add("")
    add("| Indicator | Stored ID | Main normalized column |")
    add("|---|---|---|")
    for row in civic:
        add(f"| {md_escape(row['name'])} | `{row['id']}` | `{row['id']}Norm` |")
    add("")
    add("## RAI indicators")
    add("")
    add(f"The current dictionary contains **{len(rai)} named indicators** grouped into six themes. RAI is reported separately for China, Russia, and a combined series.")
    add("")
    add("| Indicator | Stored ID | Theme |")
    add("|---|---|---|")
    for row in rai:
        add(f"| {md_escape(row['name'])} | `{row['id']}` | {md_escape(row['theme_name'])} |")
    add("")
    add("## MLEED indicators")
    add("")
    add("MLEED does not yet have a separate variable dictionary in this repository. The following names are extracted directly from the current combined output schema and therefore need substantive review before this guide is made authoritative.")
    add("The active export also contains `NormShockCountGt3` variants, which identify shocks meeting the pipeline's count threshold.")
    add("")
    add("| Output field | Normalized column | Shock column |")
    add("|---|---|---|")
    for indicator in mleed:
        add(f"| {md_escape(indicator)} | `{indicator}Norm` | `{indicator}NormShock` |")
    add("")
    add(f"Unclassified fields present in the current output: {', '.join(f'`{value}`' for value in mleed_unclassified)}.")
    add("")
    add("## News-source coverage")
    add("")
    add("Source coverage below is derived from `data/0-civic-by-source/*.csv`, using months in which `total_articles > 0`. The Civic Space and RAI pipelines share this corpus inventory, but the repository does not contain an equivalent MLEED source-level extract. Treat these as observed source records, not a promise of complete archives.")
    add("")
    unique_sources = {period.source for period in coverage.values()}
    active_pairs = [period for period in coverage.values() if period.dates]
    add(f"The extract contains **{len(unique_sources)} unique source domains**, **{len(active_pairs):,} active country-source pairs**, and source-level files for **{len(source_countries)} countries**.")
    add("")
    add(f"The source type is assigned from `build_data/constants.R`, which currently defines {international_count} international sources: international first, regional second, and otherwise local. A domain may play more than one role in project configuration; the generated table uses this precedence only for concise reporting.")
    add("")
    add("A machine-readable inventory is available as [`source_coverage.csv`](data_coverage/source_coverage.csv). Ownership and bias coding materials are described in [`source_metadata/README.md`](../source_metadata/README.md).")
    add("")
    by_country: dict[str, list[SourcePeriod]] = defaultdict(list)
    for period in coverage.values():
        by_country[period.country].append(period)
    add("### Source summary by country")
    add("")
    add("| Country | ISO3 | International | Regional | Local | Total |")
    add("|---|---|---:|---:|---:|---:|")
    for country in countries:
        periods = [period for period in by_country.get(country, []) if period.dates]
        counts = defaultdict(int)
        for period in periods:
            counts[source_types[period.source]] += 1
        add(f"| {md_escape(country)} | `{ISO3.get(country, '—')}` | {counts['International']} | {counts['Regional']} | {counts['Local']} | {len(periods)} |")
    add("")
    add("### Detailed source inventory")
    add("")
    add("Expand a country to view each observed source and its active range in the current Civic Space source-level export.")
    add("")
    for country in countries:
        periods = sorted(
            by_country.get(country, []),
            key=lambda period: (source_types[period.source], period.source),
        )
        active_count = sum(bool(period.dates) for period in periods)
        add("<details>")
        add(f"<summary><strong>{md_escape(country)} ({ISO3.get(country, 'unmapped')})</strong> — {active_count} active sources</summary>")
        add("")
        if not periods:
            add("No Civic/RAI source-level file is present in the current repository export.")
            add("")
        else:
            add("| Source | Type | Active range | Active months | Articles |")
            add("|---|---|---|---:|---:|")
            for period in periods:
                add(
                    f"| `{period.source}` | {source_types[period.source]} | "
                    f"{month_label(period.start)}–{month_label(period.end)} | "
                    f"{len(period.dates)} | {integer_label(period.total_articles)} |"
                )
            add("")
        add("</details>")
        add("")
    add("## Refreshing this guide")
    add("")
    add("From the repository root, run:")
    add("")
    add("```bash")
    add("python3 scripts/generate_data_coverage.py")
    add("```")
    add("")
    add("Review the four flags near the top after every refresh. The script also regenerates the country and source CSV inventories.")
    add("")
    return "\n".join(lines)


def main() -> None:
    summaries = {name: read_dataset(name, path) for name, path in DATASETS.items()}
    countries = sorted(
        set().union(*(summary.countries.keys() for summary in summaries.values()))
    )
    civic = read_indicator_dictionary(DATA / "cs_vars.csv")
    rai = read_indicator_dictionary(DATA / "rai_vars.csv")
    mleed, mleed_unclassified = mleed_indicators(summaries["MLEED"].fields)
    coverage = read_source_coverage()
    constants = ROOT / "build_data" / "constants.R"
    international = read_r_character_vector(constants, "isources")
    regional = read_r_character_vector(constants, "rsources")
    source_types = {
        period.source: source_type(period.source, international, regional)
        for period in coverage.values()
    }

    write_country_csv(summaries, countries)
    write_source_csv(coverage, source_types)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        build_markdown(
            summaries,
            countries,
            civic,
            rai,
            mleed,
            mleed_unclassified,
            coverage,
            source_types,
            len(international),
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    print(f"Wrote {COUNTRY_CSV.relative_to(ROOT)}")
    print(f"Wrote {SOURCE_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
