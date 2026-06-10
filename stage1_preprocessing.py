"""
================================================================================
STAGE 1 — Data Extraction & Preprocessing
SEC 10-K Filing Classification Project
================================================================================
Dataset  : winterForestStump/10-K_sec_filings (HuggingFace)
Target   : Financial risk level — high / medium / low
           (derived from Risk Factors section tone & keyword density)

Pipeline steps
──────────────
  1. Load dataset from HuggingFace Hub
  2. Inspect structure (splits, columns, dtypes, sample rows)
  3. Extract raw text from the relevant column
  4. Clean text  (HTML → boilerplate → symbols → whitespace → lowercase)
  5. Segment into 4 sections via regex keyword matching
  6. Persist result as a labelled DataFrame  (CSV + Pickle)
  7. Sanity-check output

Requirements
──────────────
  pip install datasets pandas scikit-learn
"""

# ── Standard library ──────────────────────────────────────────────────────────
import re
import os
import sys
import logging
import warnings

from typing import Any, Dict, Optional, Tuple

warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd

# HuggingFace datasets are imported lazily so demo mode still works when the
# optional package is not installed.
from typing import Any

# ── NLTK assets (downloaded once, cached after that) ──────────────────────────
# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DATASET_NAME = "winterForestStump/10-K_sec_filings"

# Output paths
OUT_DIR     = "stage1_output"
CSV_PATH    = os.path.join(OUT_DIR, "sec_filings_stage1.csv")
PICKLE_PATH = os.path.join(OUT_DIR, "sec_filings_stage1.pkl")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Boilerplate phrases to strip verbatim ─────────────────────────────────────
BOILERPLATE_PHRASES = [
    r"table\s+of\s+contents",
    r"forward[- ]looking\s+statements?",
    r"this\s+annual\s+report\s+on\s+form\s+10-?k",
    r"incorporated\s+herein\s+by\s+reference",
    r"see\s+notes?\s+to\s+consolidated\s+financial\s+statements",
    r"see\s+accompanying\s+notes",
    r"page\s+\d+\s+of\s+\d+",
    r"f-\d+",                          # page markers like F-1, F-12
    r"exhibit\s+\d+[\.\d]*",
    r"as\s+of\s+december\s+31",        # common date boilerplate
    r"pursuant\s+to\s+section\s+\d+",
    r"securities\s+exchange\s+act",
]
BOILERPLATE_RE = re.compile(
    "|".join(BOILERPLATE_PHRASES),
    re.IGNORECASE,
)

# ── Section boundary patterns ─────────────────────────────────────────────────
#   Each entry: (section_name, list_of_start_patterns)
#   Patterns match the ITEM header (SEC filings always use "ITEM N[A]. Title").
#   We capture from the matched header to the next ITEM header or end-of-text.

SECTION_PATTERNS: Dict[str, list] = {

    "risk_factors": [
        # Most filings: ITEM 1A
        r"item\s*1\s*a[\.\s]*[:\-]?\s*risk\s+factors?",
        # Some older filings label it differently
        r"risk\s+factors?\s*(?:and\s+uncertainties)?",
    ],

    "business_overview": [
        # ITEM 1 (comes before 1A, so must be matched carefully)
        r"item\s*1[\.\s]*[:\-]?\s*business\b",
        r"business\s+overview",
        r"overview\s+of\s+(?:our\s+)?business",
    ],

    "mdna": [
        # ITEM 7
        r"item\s*7[\.\s]*[:\-]?\s*management[\'\s]*s?\s+discussion",
        r"management[\'\s]*s?\s+discussion\s+and\s+analysis",
        r"md\s*[&and]+\s*a\b",
    ],

    "financial_statements": [
        # ITEM 8
        r"item\s*8[\.\s]*[:\-]?\s*financial\s+statements?",
        r"consolidated\s+(?:balance\s+sheets?|statements?\s+of)",
        r"financial\s+statements?\s+and\s+supplementary\s+data",
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Load dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_sec_dataset(name: str = DATASET_NAME):
    """
    Load the HuggingFace dataset.
    Returns (dataset_dict, text_column_name).
    Raises RuntimeError with a clear message if the Hub is unreachable.
    """
    log.info("Loading dataset: %s", name)
    try:
        from datasets import load_dataset

        ds = load_dataset(name)

        ds = ds["train"].select(range(3000))
    except Exception as exc:
        raise RuntimeError(
            f"\n[ERROR] Could not load '{name}' from HuggingFace Hub.\n"
            "  - Install the optional dependency with: pip install datasets\n"
            "  • Check your internet connection.\n"
            "  • Confirm the dataset exists at https://huggingface.co/datasets/winterForestStump/10-K_sec_filings\n"
            "  • If the dataset requires authentication, run: huggingface-cli login\n"
            f"  Original error: {exc}"
        ) from exc

    log.info("Dataset loaded successfully.")
    return ds


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Inspect structure
# ══════════════════════════════════════════════════════════════════════════════
def inspect_dataset(ds):
    print("\n" + "═"*70)
    print("DATASET INSPECTION")
    print("═"*70)

    print(f"Rows: {len(ds)}")

    print("\nColumns")
    print("-"*50)

    for col, feat in ds.features.items():
        print(f"{col:25s} {feat}")

    text_col = None

    priority = ["text", "content", "filing", "document", "raw_text", "body"]

    for p in priority:
        if p in ds.features:
            text_col = p
            break

    if text_col is None:
        raise ValueError("Could not detect text column.")

    print("\nText column:", text_col)

    return text_col

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Extract raw texts
# ══════════════════════════════════════════════════════════════════════════════

def extract_raw_texts(ds, text_col):

    log.info("Extracting raw texts...")

    texts = ds[text_col]

    raw_series = pd.Series(texts, name="raw_text", dtype=str)

    raw_series = raw_series.dropna()

    raw_series = raw_series.loc[
        raw_series.str.strip() != ""
    ]

    raw_series = raw_series.reset_index(drop=True)

    log.info("Documents after cleaning: %d", len(raw_series))

    return raw_series

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Text cleaning
# ══════════════════════════════════════════════════════════════════════════════

def remove_html(text: str) -> str:
    """Strip all HTML / XML tags."""
    # Remove script/style blocks first (they often contain code noise)
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text,
                  flags=re.IGNORECASE | re.DOTALL)
    # Remove remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    entities = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&nbsp;": " ", "&quot;": '"', "&#39;": "'",
        "&ldquo;": '"', "&rdquo;": '"', "&mdash;": "—",
        "&ndash;": "–", "&lsquo;": "'", "&rsquo;": "'",
    }
    for ent, char in entities.items():
        text = text.replace(ent, char)
    # Remove any residual &#123; style entities
    text = re.sub(r"&#?\w+;", " ", text)
    return text


def remove_boilerplate(text: str) -> str:
    """Remove known boilerplate phrases (case-insensitive)."""
    return BOILERPLATE_RE.sub(" ", text)


def remove_special_symbols(text: str) -> str:
    """
    Remove noise characters while preserving:
      - Letters, digits, spaces
      - Punctuation useful for financial text (% $ , . ; : ' " - / ( ) )
    """
    # Remove XBRL / inline XBRL tags  e.g.  ix:nonfraction
    text = re.sub(r"\bix:[a-z]+\b", " ", text, flags=re.IGNORECASE)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove e-mail addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    # Remove hex colour codes / technical ids
    text = re.sub(r"\b[0-9a-fA-F]{6,}\b", " ", text)
    # Remove standalone numbers that are pure noise (e.g. page numbers alone)
    # Keep numbers that are part of words or have $ / % context
    text = re.sub(r"(?<!\w)(\d{1,3}(?:,\d{3})*|\d+)(?!\s*[%$,.\w])", " ", text)
    # Strip characters that are not alphanumeric or useful punctuation
    text = re.sub(r"[^\w\s\$\%\.\,\;\:\'\"\-\/\(\)]", " ", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse all runs of whitespace (spaces, tabs, newlines) to a single space."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(raw: str) -> str:
    """
    Master cleaning function.
    Order matters:  HTML  →  boilerplate  →  symbols  →  whitespace  →  lowercase
    """
    if not isinstance(raw, str) or not raw.strip():
        return ""

    text = remove_html(raw)
    text = remove_boilerplate(text)
    text = remove_special_symbols(text)
    text = normalize_whitespace(text)
    text = text.lower()           # lowercase AFTER symbol removal (preserves ITEM N pattern for segmentation)
    return text


def clean_text_preserve_case(raw: str) -> str:
    """
    Same as clean_text but keeps original casing.
    Used for section segmentation (regex patterns are case-insensitive but
    casing in the text helps verify matches).
    """
    if not isinstance(raw, str) or not raw.strip():
        return ""
    text = remove_html(raw)
    text = remove_boilerplate(text)
    text = remove_special_symbols(text)
    text = normalize_whitespace(text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Section segmentation
# ══════════════════════════════════════════════════════════════════════════════

def build_section_regex(patterns: list) -> re.Pattern:
    """
    Combine a list of pattern strings into one compiled regex.
    Each alternative is wrapped in a non-capturing group.
    """
    combined = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(combined, re.IGNORECASE)


# Pre-compile all section patterns
_SECTION_RE: Dict[str, re.Pattern] = {
    name: build_section_regex(pats)
    for name, pats in SECTION_PATTERNS.items()
}

# Compile a single "any ITEM header" pattern to detect section boundaries
_ANY_ITEM_RE = re.compile(
    r"\bitem\s*\d+\s*[a-c]?\s*[\.\:\-]",
    re.IGNORECASE,
)


def _find_section_span(text: str, section_re: re.Pattern) -> Optional[Tuple[int, int]]:
    """
    Locate the span of a section within `text`.

    Strategy
    ────────
    1. Find the first match of the section's header regex.
    2. The section content starts right after that match.
    3. The section ends at the next ITEM header or end-of-text.

    Returns (start, end) character positions, or None if not found.
    """
    header_match = section_re.search(text)
    if not header_match:
        return None

    content_start = header_match.end()

    # Find next ITEM header after content_start
    next_item = _ANY_ITEM_RE.search(text, content_start)

    # Make sure "next_item" is not just the same match again (edge case where
    # the header itself contains "ITEM")
    if next_item and next_item.start() <= header_match.start() + 5:
        next_item = _ANY_ITEM_RE.search(text, next_item.end())

    content_end = next_item.start() if next_item else len(text)
    return (content_start, content_end)


def extract_sections(text: str) -> Dict[str, str]:
    """
    Extract all four target sections from a single filing text.

    Parameters
    ----------
    text : str
        Case-preserved, HTML-stripped, whitespace-normalised filing text.

    Returns
    -------
    dict with keys: risk_factors, business_overview, mdna, financial_statements
    Each value is the cleaned section text, or "" if the section was not found.
    """
    results: Dict[str, str] = {}

    for section_name, section_re in _SECTION_RE.items():
        span = _find_section_span(text, section_re)
        if span:
            raw_section = text[span[0]: span[1]]
            # Clean the extracted section (lowercase, symbol strip)
            cleaned = clean_text(raw_section)
            # Truncate very long sections to keep memory reasonable
            # 10,000 words ≈ ~70,000 chars; most sections are shorter
            results[section_name] = cleaned[:70_000]
        else:
            results[section_name] = ""

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Build DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def build_dataframe(raw_texts: pd.Series) -> pd.DataFrame:
    """
    For each filing:
      1. Clean (preserve case) → use for segmentation
      2. Segment into 4 sections
      3. Also store the full cleaned+lowercased text

    Returns a DataFrame with columns:
      raw_text, cleaned_text, risk_factors, business_overview,
      mdna, financial_statements, section_coverage
    """
    log.info("Building DataFrame for %d filings …", len(raw_texts))

    records = []
    for idx, raw in enumerate(raw_texts):
        if idx % 100 == 0 and idx > 0:
            log.info("  Processed %d / %d …", idx, len(raw_texts))

        # Case-preserved clean (for segmentation)
        clean_cased = clean_text_preserve_case(raw)
        # Fully lowercased clean (for features later)
        clean_lower = clean_text(raw)
        # Section extraction
        sections    = extract_sections(clean_cased)

        # Coverage metric: how many sections were found (0–4)
        found_count = sum(1 for v in sections.values() if v.strip())

        records.append({
            "raw_text"            : raw,
            "cleaned_text"        : clean_lower,
            "risk_factors"        : sections["risk_factors"],
            "business_overview"   : sections["business_overview"],
            "mdna"                : sections["mdna"],
            "financial_statements": sections["financial_statements"],
            "section_coverage"    : found_count,          # 0-4
        })

    df = pd.DataFrame(records)
    log.info("DataFrame built: %s", str(df.shape))
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — Sanity checks & report
# ══════════════════════════════════════════════════════════════════════════════

def print_report(df: pd.DataFrame) -> None:
    """Print a comprehensive quality report for the processed DataFrame."""

    section_cols = ["risk_factors", "business_overview", "mdna", "financial_statements"]

    print("\n" + "═" * 70)
    print("  STAGE 1 OUTPUT REPORT")
    print("═" * 70)

    print(f"\n  DataFrame shape : {df.shape}")
    print(f"  Columns         : {list(df.columns)}\n")

    # ── Section extraction rates ───────────────────────────────────────────────
    print("  Section extraction rates (% of filings where section was found)")
    print("  " + "─" * 56)
    for col in section_cols:
        n_found = (df[col].str.strip() != "").sum()
        pct     = n_found / len(df) * 100
        bar     = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {col:24s}  [{bar}] {pct:5.1f}%  ({n_found:,}/{len(df):,})")

    # ── Coverage distribution ──────────────────────────────────────────────────
    print("\n  Section coverage distribution (how many sections found per filing)")
    cov_counts = df["section_coverage"].value_counts().sort_index()
    for n_sec, count in cov_counts.items():
        print(f"    {n_sec} sections found : {count:,} filings")

    # ── Text length stats ──────────────────────────────────────────────────────
    print("\n  Cleaned text length (characters)")
    lengths = df["cleaned_text"].str.len()
    print(f"    min    : {lengths.min():,}")
    print(f"    median : {int(lengths.median()):,}")
    print(f"    mean   : {int(lengths.mean()):,}")
    print(f"    max    : {lengths.max():,}")

    # ── Risk Factors section length stats ─────────────────────────────────────
    rf_lengths = df["risk_factors"].str.len()
    rf_nonzero = rf_lengths[rf_lengths > 0]
    print(f"\n  Risk Factors section length (non-empty, chars)")
    if len(rf_nonzero):
        print(f"    min    : {rf_nonzero.min():,}")
        print(f"    median : {int(rf_nonzero.median()):,}")
        print(f"    max    : {rf_nonzero.max():,}")

    # ── Sample record ──────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  SAMPLE RECORD (index 0)")
    print("─" * 70)
    row = df.iloc[0]
    print(f"\n  cleaned_text (first 300 chars):\n  {row['cleaned_text'][:300]}")
    for col in section_cols:
        snippet = row[col][:200] if row[col] else "[NOT FOUND]"
        print(f"\n  {col} (first 200 chars):\n  {snippet}")
    print(f"\n  section_coverage : {row['section_coverage']} / 4")

    print("\n" + "═" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS — Offline / demo mode
# ══════════════════════════════════════════════════════════════════════════════

DEMO_FILINGS = [
    # High-risk filing (dense negative language in risk factors)
    {
        "text": """
        <html><body>
        <b>Table of Contents</b>
        <p>ITEM 1. BUSINESS</p>
        <p>TechCorp Inc. is a publicly traded technology company providing enterprise
        software solutions to Fortune 500 clients since 2005. We operate in North America,
        Europe, and Asia Pacific markets. Our products include cloud ERP, analytics platforms,
        and cybersecurity tooling. Revenue for fiscal 2023 was approximately $1.2 billion.</p>

        <p>ITEM 1A. RISK FACTORS</p>
        <p>Our business faces numerous material risks and uncertainties. We have identified
        a material weakness in our internal control over financial reporting related to
        our revenue recognition procedures. This material weakness could result in
        misstatements in our financial statements. We are currently subject to significant
        litigation including class action lawsuits from shareholders alleging securities fraud.
        We face substantial risk of default on our senior secured credit facility of $800M
        due to covenant violations. Liquidity constraints may force asset impairment charges.
        Bankruptcy proceedings cannot be ruled out if capital markets remain inaccessible.
        Regulatory investigations by the SEC into our accounting practices pose existential risk.
        Our debt-to-equity ratio of 4.2x exceeds industry norms significantly. Supply chain
        disruptions, geopolitical instability, and inflationary pressures have materially
        impaired our gross margins by 800 basis points year-over-year.</p>

        <p>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION</p>
        <p>Revenue declined 18% year-over-year to $1.2B from $1.46B driven by customer churn
        and contract renegotiations. Gross margin compressed to 34% from 42% prior year.
        Operating losses of $230M were recorded. Cash burn rate of $45M per quarter raises
        going-concern questions. We have drawn down $600M of our revolving credit facility.
        Management continues to evaluate strategic alternatives including asset sales.</p>

        <p>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</p>
        <p>Consolidated Balance Sheet: Total assets $2.1B, Total liabilities $1.9B,
        Stockholders equity $200M. Net loss $310M. Operating cash flow negative $180M.
        Long-term debt $1.4B. Goodwill impairment charge $95M recognized in Q4.</p>
        </body></html>
        """
    },
    # Low-risk filing (stable, positive language)
    {
        "text": """
        <html><body>
        <b>Table of Contents</b>
        <p>ITEM 1. BUSINESS OVERVIEW</p>
        <p>HealthPlus Corp is a leading managed care organization serving 4.2 million members
        across 12 states. Founded in 1989, we have maintained consistent profitability for
        28 consecutive years. Our diversified revenue model spans commercial insurance,
        Medicare Advantage, and Medicaid managed care. We employ approximately 18,000
        dedicated healthcare professionals committed to member outcomes.</p>

        <p>ITEM 1A. RISK FACTORS</p>
        <p>Our business is subject to normal risks inherent in the managed healthcare industry.
        Changes in government reimbursement rates could modestly affect margins. Competition
        from new market entrants may require incremental investment in technology and service
        differentiation. We maintain robust reserves above regulatory minimums and our risk-
        based capital ratio of 320% substantially exceeds statutory requirements. Our
        investment portfolio is conservatively positioned in high-grade fixed income
        securities with minimal credit risk. We continuously monitor regulatory developments
        and maintain constructive relationships with state insurance departments.</p>

        <p>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</p>
        <p>Revenue grew 9.2% year-over-year to $8.4B driven by membership growth and
        rate improvements. Medical loss ratio of 82.1% improved 60bps versus prior year.
        Operating income increased 14% to $680M. We returned $340M to shareholders
        via dividends and buybacks. Balance sheet remains fortress-like with $2.1B cash
        and no near-term debt maturities. Full-year guidance raised to $7.80-8.00 EPS.</p>

        <p>ITEM 8. FINANCIAL STATEMENTS</p>
        <p>Total revenue $8.4B. Net income $510M. Total assets $12.3B. Long-term debt $1.8B.
        Shareholders equity $4.2B. Return on equity 12.4%. Dividend per share $2.40.
        Book value per share $48.20. Debt-to-capital ratio 30%. AA- credit rating maintained.</p>
        </body></html>
        """
    },
    # Medium-risk filing
    {
        "text": """
        <html><body>
        <p>ITEM 1. BUSINESS</p>
        <p>RetailChain Inc. operates 1,400 specialty retail stores across the United States
        and Canada selling consumer electronics and home appliances. We compete primarily
        with big-box retailers and e-commerce platforms. Founded in 1978, we have adapted
        our business model through multiple industry cycles. Annual revenues approximate
        $6.8 billion with EBITDA margins around 7-9%.</p>

        <p>ITEM 1A. RISK FACTORS</p>
        <p>We face meaningful competition from online retailers that may pressure our pricing
        and traffic trends. Consumer spending patterns are sensitive to macroeconomic conditions
        including interest rates and employment levels. Our lease obligations represent a
        significant fixed cost base requiring consistent revenue generation. Vendor
        concentration creates some dependency risk with our top three suppliers accounting
        for 38% of inventory purchases. Cybersecurity threats to our point-of-sale systems
        represent an ongoing operational risk we actively manage. Tariff uncertainties on
        imported goods could increase product costs. We maintain adequate insurance coverage
        and business continuity plans for these operational risks.</p>

        <p>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</p>
        <p>Comparable store sales increased 1.8% year-over-year, slightly below our 2-3%
        long-term target. Total revenue grew 3.1% to $6.8B aided by new store openings.
        Gross margin of 28.4% was flat versus prior year. SG&A leverage improved 20bps.
        Operating income of $410M was in line with guidance. Free cash flow of $285M
        funded $120M in share repurchases and maintained our quarterly dividend.</p>

        <p>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</p>
        <p>Total revenue $6.8B. Net income $210M. Total assets $4.1B. Long-term debt $900M.
        Lease liabilities $2.2B. Shareholders equity $680M. EPS $3.42. Dividend $0.80/share.
        Interest coverage ratio 4.8x. Debt-to-EBITDA 1.7x within covenant limits.</p>
        </body></html>
        """
    },
]

from datasets import Dataset, DatasetDict
def make_demo_dataset() -> DatasetDict:
    """
    Build a tiny in-memory DatasetDict that mirrors the real dataset schema,
    used when the Hub cannot be reached (e.g. network-restricted environment).
    """
    import random
    # Expand to 30 synthetic records by cycling through templates
    expanded = []
    for i in range(30):
        base   = DEMO_FILINGS[i % len(DEMO_FILINGS)].copy()
        # Add slight variation so they're not identical
        base["text"] += f"\n<!-- filing_id={i} -->"
        expanded.append(base)

    hf_dataset = Dataset.from_list(expanded)
    return DatasetDict({"train": hf_dataset})


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  SEC 10-K CLASSIFICATION PROJECT  —  STAGE 1: PREPROCESSING")
    print("=" * 70 + "\n")

    # ── STEP 1: Load dataset ──────────────────────────────────────────────────
    try:
        ds = load_sec_dataset(DATASET_NAME)
        demo_mode = False
    except RuntimeError as e:
        log.warning("%s", e)
        log.warning("Falling back to DEMO MODE with synthetic 10-K filings.")
        log.warning("On your machine with internet access, replace this with "
                    "load_dataset('%s').", DATASET_NAME)
        ds = make_demo_dataset()
        demo_mode = True

    # ── STEP 2: Inspect ───────────────────────────────────────────────────────
    text_col = inspect_dataset(ds)

    # ── STEP 3: Extract raw texts ─────────────────────────────────────────────
    raw_texts = extract_raw_texts(ds, text_col)

    # ── STEPS 4 + 5 + 6: Clean, segment, build DataFrame ─────────────────────
    df = build_dataframe(raw_texts)

    # ── STEP 7: Quality report ────────────────────────────────────────────────
    print_report(df)

    # ── Persist outputs ───────────────────────────────────────────────────────
    df.to_csv(CSV_PATH, index=False)
    df.to_pickle(PICKLE_PATH)
    log.info("Saved CSV   → %s", CSV_PATH)
    log.info("Saved Pickle→ %s", PICKLE_PATH)

    if demo_mode:
        print("\n" + "⚠" * 35)
        print("  DEMO MODE — Replace make_demo_dataset() with the real Hub call")
        print("  when running on a machine with internet access to HuggingFace.")
        print("⚠" * 35 + "\n")
    else:
        print("\n✅  Stage 1 complete. Outputs saved to:", OUT_DIR)

    return df


if __name__ == "__main__":
    df = main()
