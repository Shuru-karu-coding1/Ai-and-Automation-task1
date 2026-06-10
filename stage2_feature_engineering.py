"""
================================================================================
STAGE 2 — Feature Engineering & Label Generation
SEC 10-K Filing Classification Project
================================================================================
Input   : stage1_output/sec_filings_stage1.pkl  (produced by stage1_preprocessing.py)
Outputs : stage2_output/
            ├── sec_filings_stage2.pkl      – full DataFrame with labels & features
            ├── X_train.pkl / X_test.pkl    – TF-IDF sparse matrices
            ├── y_train.pkl / y_test.pkl    – encoded label arrays
            ├── tfidf_vectorizer.joblib     – fitted TF-IDF vectorizer
            ├── label_encoder.joblib        – fitted LabelEncoder
            └── stage2_report.txt           – human-readable summary

Classification target
─────────────────────
  Financial risk level:  high | medium | low
  Derived from: Risk Factors section keyword density + MDNA section sentiment signals.

  Justification
  ─────────────
  The Risk Factors section is the primary source because SEC regulations (Item 1A)
  require companies to disclose ALL material risks.  High-risk companies use more
  severe, urgent language (default, bankruptcy, impairment) while low-risk companies
  use hedged, stable language (conservative, robust, exceeds requirements).
  MDNA adds financial outcome signals (declining revenue, negative cash flow)
  that reinforce the risk assessment beyond just stated risks.

Requirements
────────────
  pip install scikit-learn joblib pandas imbalanced-learn
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import logging
import warnings
import textwrap
from io import StringIO
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════

STAGE1_PICKLE = "stage1_output/sec_filings_stage1.pkl"
OUT_DIR       = "stage2_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Output file paths ─────────────────────────────────────────────────────────
STAGE2_PICKLE     = os.path.join(OUT_DIR, "sec_filings_stage2.pkl")
X_TRAIN_PATH      = os.path.join(OUT_DIR, "X_train.pkl")
X_TEST_PATH       = os.path.join(OUT_DIR, "X_test.pkl")
Y_TRAIN_PATH      = os.path.join(OUT_DIR, "y_train.pkl")
Y_TEST_PATH       = os.path.join(OUT_DIR, "y_test.pkl")
VECTORIZER_PATH   = os.path.join(OUT_DIR, "tfidf_vectorizer.joblib")
LABEL_ENC_PATH    = os.path.join(OUT_DIR, "label_encoder.joblib")
REPORT_PATH       = os.path.join(OUT_DIR, "stage2_report.txt")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — KEYWORD DICTIONARIES (the heart of the labelling logic)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Design principles
#  ─────────────────
#  1. Each keyword is a plain substring (lowercased) — no regex overhead.
#  2. Keywords are grouped by SEVERITY within each risk tier so weights can be
#     tuned independently.
#  3. A MODIFIER dictionary dampens or amplifies scores when context words
#     appear nearby (e.g. "not in default" should not score as high-risk).
#  4. Positive signals are scored separately and subtracted from the final
#     score to capture net risk sentiment.

# ── High-risk keyword groups ──────────────────────────────────────────────────
HIGH_RISK_KEYWORDS: Dict[str, List[str]] = {

    # Existential / severe (weight ×3 each)
    "existential": [
        "bankruptcy",
        "insolvency",
        "going concern",
        "chapter 11",
        "chapter 7",
        "cease operations",
        "liquidation",
        "dissolution",
    ],

    # Financial distress (weight ×2 each)
    "financial_distress": [
        "default",
        "covenant violation",
        "covenant breach",
        "debt covenant",
        "cross-default",
        "acceleration of debt",
        "material weakness",
        "restatement",
        "going-concern",
        "impairment",
        "goodwill impairment",
        "asset impairment",
        "write-down",
        "write-off",
        "negative cash flow",
        "cash burn",
        "liquidity constraint",
        "working capital deficit",
        "stockholders deficit",
    ],

    # Legal & regulatory (weight ×2 each)
    "legal_regulatory": [
        "litigation",
        "class action",
        "securities fraud",
        "regulatory investigation",
        "sec investigation",
        "doj investigation",
        "subpoena",
        "criminal investigation",
        "indictment",
        "fraud",
        "whistleblower",
        "sanctions violation",
    ],

    # Operational stress (weight ×1 each)
    "operational_stress": [
        "material adverse",
        "significant uncertainty",
        "substantial doubt",
        "significant risk",
        "critical risk",
        "existential threat",
        "unable to meet",
        "inability to",
        "operational disruption",
        "supply chain disruption",
        "customer concentration",
        "vendor concentration",
        "single customer",
        "key personnel",
        "loss of key",
        "cybersecurity breach",
        "data breach",
        "ransomware",
        "significant decline",
        "declining revenue",
        "margin compression",
        "operating losses",
        "net losses",
        "accumulated deficit",
        "significant competition",
    ],
}

# ── Low-risk keyword groups ────────────────────────────────────────────────────
LOW_RISK_KEYWORDS: Dict[str, List[str]] = {

    # Financial strength (weight ×3 each)
    "financial_strength": [
        "investment grade",
        "strong balance sheet",
        "fortress balance sheet",
        "no debt",
        "debt free",
        "cash rich",
        "exceeds regulatory",
        "above regulatory minimum",
        "risk-based capital",
        "well capitalized",
        "aa rated",
        "aaa rated",
    ],

    # Stability signals (weight ×2 each)
    "stability": [
        "consistently profitable",
        "consecutive years",
        "stable revenue",
        "recurring revenue",
        "long-term contracts",
        "diversified revenue",
        "conservative",
        "robust reserves",
        "adequate insurance",
        "business continuity",
        "hedging program",
        "fully hedged",
        "strong cash flow",
        "free cash flow positive",
        "dividend growth",
        "share repurchase",
    ],

    # Positive operational (weight ×1 each)
    "positive_operational": [
        "market leader",
        "competitive advantage",
        "strong customer",
        "customer retention",
        "low churn",
        "revenue growth",
        "margin expansion",
        "cost reduction",
        "efficiency improvement",
        "technological advantage",
        "intellectual property",
        "patent portfolio",
        "regulatory approval",
        "compliance program",
        "internal controls",          # positive when standalone (not preceded by "weakness in")
        "risk management framework",
        "enterprise risk management",
    ],
}

# ── Negation / context modifiers ──────────────────────────────────────────────
#  If any of these words appear within 60 characters BEFORE a high-risk keyword,
#  that keyword's contribution is halved (we detected negation or mitigation).
NEGATION_CONTEXT = [
    "not", "no ", "without", "mitigated", "managed", "reduced",
    "historically", "did not", "have not", "has not", "no material",
    "adequately", "insurance covers", "fully insured",
]

# ── Score weights ─────────────────────────────────────────────────────────────
WEIGHTS: Dict[str, int] = {
    # high-risk groups
    "existential":        3,
    "financial_distress": 2,
    "legal_regulatory":   2,
    "operational_stress": 1,
    # low-risk groups
    "financial_strength": 3,
    "stability":          2,
    "positive_operational": 1,
}

# ── Thresholds for label assignment ───────────────────────────────────────────
#
#  net_score = high_score - low_score
#
#  net_score ≥  HIGH_THRESHOLD  →  "high"
#  net_score ≤  LOW_THRESHOLD   →  "low"
#  in between                   →  "medium"
#
HIGH_THRESHOLD =  4     # at least 4 net-high-risk points
LOW_THRESHOLD  = -3     # at least 3 net-low-risk points

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 (cont.) — Scoring & labelling functions
# ══════════════════════════════════════════════════════════════════════════════

def _count_keyword_hits(
    text: str,
    keyword_groups: Dict[str, List[str]],
    check_negation: bool = False,
) -> Tuple[float, Dict[str, int]]:
    """
    Count weighted keyword hits in `text`.

    Parameters
    ----------
    text            : lower-cased document text
    keyword_groups  : dict of {group_name: [keyword, ...]}
    check_negation  : if True, halve score when negation context precedes keyword

    Returns
    -------
    total_score     : float — sum of weighted, optionally negation-adjusted hits
    detail          : dict of {group_name: raw_hit_count}
    """
    total_score = 0.0
    detail: Dict[str, int] = {}

    for group_name, keywords in keyword_groups.items():
        weight    = WEIGHTS.get(group_name, 1)
        hit_count = 0

        for kw in keywords:
            pos = 0
            while True:
                idx = text.find(kw, pos)
                if idx == -1:
                    break
                hit_count += 1
                contribution = weight

                # Check for negation in the 60-char window before the keyword
                if check_negation and idx > 0:
                    window = text[max(0, idx - 60): idx]
                    if any(neg in window for neg in NEGATION_CONTEXT):
                        contribution *= 0.5   # halve — mitigated risk

                total_score += contribution
                pos = idx + len(kw)

        detail[group_name] = hit_count

    return total_score, detail


def compute_risk_score(risk_text: str, mdna_text: str = "") -> Dict:
    """
    Compute a numeric risk score for a single filing.

    Uses risk_factors as the primary signal (full weight).
    MDNA is used as a secondary signal at 50% weight because it reflects
    *outcomes* rather than *stated risks*, making it a corroborating signal.

    Returns a dict with:
      high_score, low_score, net_score, label, detail_high, detail_low
    """
    text_rf   = risk_text.lower()  if risk_text  else ""
    text_mdna = mdna_text.lower()  if mdna_text  else ""

    # ── High-risk scores ───────────────────────────────────────────────────────
    hs_rf,   dh_rf   = _count_keyword_hits(text_rf,   HIGH_RISK_KEYWORDS, check_negation=True)
    hs_mdna, dh_mdna = _count_keyword_hits(text_mdna, HIGH_RISK_KEYWORDS, check_negation=True)
    high_score = hs_rf + 0.5 * hs_mdna          # MDNA at half weight

    # ── Low-risk (positive) scores ─────────────────────────────────────────────
    ls_rf,   dl_rf   = _count_keyword_hits(text_rf,   LOW_RISK_KEYWORDS)
    ls_mdna, dl_mdna = _count_keyword_hits(text_mdna, LOW_RISK_KEYWORDS)
    low_score = ls_rf + 0.5 * ls_mdna

    # ── Net score and label ───────────────────────────────────────────────────
    net_score = high_score - low_score

    if net_score >= HIGH_THRESHOLD:
        label = "high"
    elif net_score <= LOW_THRESHOLD:
        label = "low"
    else:
        label = "medium"

    return {
        "high_score"  : round(high_score, 2),
        "low_score"   : round(low_score,  2),
        "net_score"   : round(net_score,  2),
        "risk_label"  : label,
        "detail_high" : {**{f"rf_{k}": v for k, v in dh_rf.items()},
                         **{f"mdna_{k}": v for k, v in dh_mdna.items()}},
        "detail_low"  : {**{f"rf_{k}": v for k, v in dl_rf.items()},
                         **{f"mdna_{k}": v for k, v in dl_mdna.items()}},
    }


def apply_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply compute_risk_score to every row and append label + diagnostic columns.
    """
    log.info("Computing risk scores and labels for %d filings …", len(df))

    results = df.apply(
        lambda row: compute_risk_score(
            risk_text = row.get("risk_factors", ""),
            mdna_text = row.get("mdna", ""),
        ),
        axis=1,
    )

    df = df.copy()
    df["risk_label"]  = results.apply(lambda r: r["risk_label"])
    df["high_score"]  = results.apply(lambda r: r["high_score"])
    df["low_score"]   = results.apply(lambda r: r["low_score"])
    df["net_score"]   = results.apply(lambda r: r["net_score"])

    log.info("Labels assigned.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Class distribution analysis
# ══════════════════════════════════════════════════════════════════════════════

def print_class_distribution(df: pd.DataFrame, report_lines: List[str]) -> None:
    """Print and record class distribution with a visual bar chart."""

    counts = df["risk_label"].value_counts()
    total  = len(df)

    header = "\n" + "═" * 60 + "\n  CLASS DISTRIBUTION\n" + "═" * 60
    print(header)
    report_lines.append(header)

    for label in ["high", "medium", "low"]:
        n   = counts.get(label, 0)
        pct = n / total * 100
        bar = "█" * int(pct / 3) + "░" * (34 - int(pct / 3))
        line = f"  {label:8s}  [{bar}]  {n:4d}  ({pct:5.1f}%)"
        print(line)
        report_lines.append(line)

    print(f"\n  Total: {total}")
    report_lines.append(f"\n  Total: {total}")

    # ── Class weight recommendation ────────────────────────────────────────────
    classes     = np.array(["high", "low", "medium"])
    y_str       = df["risk_label"].values
    y_present   = np.intersect1d(classes, y_str)
    cw          = compute_class_weight("balanced", classes=y_present, y=y_str)
    cw_dict     = dict(zip(y_present, cw.round(3)))

    advice = f"\n  Recommended class_weight for models: {cw_dict}"
    print(advice)
    report_lines.append(advice)

    # ── Score distribution ────────────────────────────────────────────────────
    print("\n  Score statistics:")
    report_lines.append("\n  Score statistics:")
    for col in ["high_score", "low_score", "net_score"]:
        stats = df[col].describe()[["min","mean","50%","max"]].round(2).to_dict()
        line  = f"  {col:12s}  min={stats['min']}  mean={stats['mean']}  median={stats['50%']}  max={stats['max']}"
        print(line)
        report_lines.append(line)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Build combined_text feature column
# ══════════════════════════════════════════════════════════════════════════════

def build_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine risk_factors + mdna into a single `combined_text` column.

    Design rationale
    ────────────────
    Risk Factors is the primary signal for risk classification (Item 1A,
    mandated by SEC). MDNA provides financial outcome context (revenue trends,
    margin direction, liquidity) that acts as a ground-truth check on whether
    the stated risks materialised. The two sections are concatenated with a
    section separator token so the TF-IDF vectorizer can still learn
    section-specific n-gram patterns if present in both sections.
    """
    log.info("Building combined_text (risk_factors + mdna) …")

    df = df.copy()

    def _combine(row: pd.Series) -> str:
        rf   = str(row.get("risk_factors", "") or "").strip()
        mdna = str(row.get("mdna", "")          or "").strip()
        parts = []
        if rf:
            parts.append(rf)
        if mdna:
            parts.append(mdna)
        return " ".join(parts)

    df["combined_text"] = df.apply(_combine, axis=1)

    # Basic stats
    lengths = df["combined_text"].str.split().str.len()
    log.info(
        "combined_text word-count stats — min: %d | mean: %.0f | max: %d",
        lengths.min(), lengths.mean(), lengths.max(),
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — TF-IDF Vectorisation
# ══════════════════════════════════════════════════════════════════════════════

def build_tfidf_features(
    texts: pd.Series,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[TfidfVectorizer, object]:
    """
    Fit a TF-IDF vectorizer on the full text corpus and return:
      (fitted_vectorizer, feature_matrix)

    Hyperparameter choices
    ──────────────────────
    max_features=5000   : keeps vocabulary manageable; SEC filings are wordy
                          but most tokens beyond rank 5000 are noise/company names.
    ngram_range=(1,2)   : bigrams capture crucial phrases like "material weakness",
                          "going concern", "class action" that unigrams miss.
    min_df=2            : ignore terms appearing in only 1 document (likely typos).
    max_df=0.95         : ignore terms in >95% of docs (too common to discriminate).
    sublinear_tf=True   : log-scaled TF dampens the effect of very frequent terms.
    strip_accents='unicode': normalise accented chars common in company names.
    """
    log.info(
        "Fitting TF-IDF: max_features=%d, ngram_range=%s, corpus_size=%d",
        max_features, ngram_range, len(texts),
    )

    vectorizer = TfidfVectorizer(
        max_features  = max_features,
        ngram_range   = ngram_range,
        min_df        = 2,           # must appear in at least 2 documents
        max_df        = 0.95,        # ignore near-universal tokens
        sublinear_tf  = True,        # apply log(1 + tf) scaling
        strip_accents = "unicode",
        analyzer      = "word",
        token_pattern = r"(?u)\b[a-z][a-z]+\b",   # alphabetic tokens only
    )

    X = vectorizer.fit_transform(texts)

    log.info(
        "TF-IDF matrix shape: %s | vocabulary size: %d",
        X.shape,
        len(vectorizer.vocabulary_),
    )
    return vectorizer, X


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Label encoding & train/test split
# ══════════════════════════════════════════════════════════════════════════════

def encode_labels(y: pd.Series) -> Tuple[LabelEncoder, np.ndarray]:
    """
    Encode string labels → integers.
    LabelEncoder maps alphabetically: high=0, low=1, medium=2
    We explicitly reorder so the mapping is intuitive for the model report.
    """
    le = LabelEncoder()
    le.fit(["high", "low", "medium"])      # fix the mapping regardless of data order
    y_encoded = le.transform(y)
    log.info("Label encoding: %s", dict(zip(le.classes_, le.transform(le.classes_))))
    return le, y_encoded


def stratified_split(
    X,
    y_encoded: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple:
    """
    80/20 stratified split.
    Stratification preserves class proportions in both train and test sets,
    which is critical when classes are imbalanced.

    Falls back to regular split if any class has fewer than 2 samples
    (can happen with very small datasets in demo mode).
    """
    from collections import Counter
    class_counts = Counter(y_encoded)
    min_count    = min(class_counts.values())

    if min_count >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size     = test_size,
            random_state  = random_state,
            stratify      = y_encoded,
        )
        log.info("Stratified 80/20 split applied.")
    else:
        log.warning(
            "Smallest class has only %d sample(s) — using non-stratified split. "
            "This is expected in demo mode; the real dataset will have sufficient samples.",
            min_count,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size    = test_size,
            random_state = random_state,
        )

    log.info(
        "Train: %d samples | Test: %d samples",
        X_train.shape[0], X_test.shape[0],
    )
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Save artefacts
# ══════════════════════════════════════════════════════════════════════════════

def save_artefacts(
    df         : pd.DataFrame,
    X_train,
    X_test,
    y_train    : np.ndarray,
    y_test     : np.ndarray,
    vectorizer : TfidfVectorizer,
    le         : LabelEncoder,
) -> None:
    """Persist all Stage 2 outputs to disk."""

    df.to_pickle(STAGE2_PICKLE)
    log.info("Saved DataFrame      → %s", STAGE2_PICKLE)

    joblib.dump(X_train, X_TRAIN_PATH)
    joblib.dump(X_test,  X_TEST_PATH)
    log.info("Saved X_train        → %s", X_TRAIN_PATH)
    log.info("Saved X_test         → %s", X_TEST_PATH)

    joblib.dump(y_train, Y_TRAIN_PATH)
    joblib.dump(y_test,  Y_TEST_PATH)
    log.info("Saved y_train        → %s", Y_TRAIN_PATH)
    log.info("Saved y_test         → %s", Y_TEST_PATH)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    log.info("Saved TF-IDF         → %s", VECTORIZER_PATH)

    joblib.dump(le, LABEL_ENC_PATH)
    log.info("Saved LabelEncoder   → %s", LABEL_ENC_PATH)


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTING helper
# ══════════════════════════════════════════════════════════════════════════════

def print_sample_scores(df: pd.DataFrame, n: int = 6) -> None:
    """Print a diagnostic table showing score breakdown for n filings."""
    print("\n" + "─" * 90)
    print(f"  SAMPLE SCORE BREAKDOWN (first {n} filings)")
    print("─" * 90)
    cols    = ["risk_label", "high_score", "low_score", "net_score"]
    display = df[cols].head(n).to_string(index=True)
    print(display)
    print("─" * 90 + "\n")


def print_tfidf_top_features(
    vectorizer : TfidfVectorizer,
    df         : pd.DataFrame,
    n_top      : int = 15,
) -> None:
    """
    Print the top TF-IDF features (highest mean TF-IDF score) per risk class.
    Helps validate that the vectorizer captures the right signals.
    """
    import scipy.sparse as sp

    feature_names = np.array(vectorizer.get_feature_names_out())
    print("\n" + "─" * 70)
    print(f"  TOP {n_top} TF-IDF FEATURES PER RISK CLASS")
    print("─" * 70)

    for label in ["high", "medium", "low"]:
        mask  = df["risk_label"] == label
        texts = df.loc[mask, "combined_text"]
        if texts.empty:
            continue
        X_sub = vectorizer.transform(texts)
        means = np.asarray(X_sub.mean(axis=0)).flatten()
        top_i = means.argsort()[::-1][:n_top]
        terms = feature_names[top_i]
        print(f"\n  [{label.upper()} RISK]  top terms by mean TF-IDF:")
        print("  " + " | ".join(terms))

    print("─" * 70 + "\n")


def print_final_summary(
    df         : pd.DataFrame,
    X_train,
    X_test,
    vectorizer : TfidfVectorizer,
    le         : LabelEncoder,
    report_lines: List[str],
) -> None:
    """Print and record the final stage summary."""
    lines = [
        "",
        "═" * 60,
        "  STAGE 2 SUMMARY",
        "═" * 60,
        f"  Total filings          : {len(df):,}",
        f"  Labelled filings       : {df['risk_label'].notna().sum():,}",
        f"  Label mapping          : {dict(zip(le.classes_, le.transform(le.classes_)))}",
        f"  TF-IDF vocabulary      : {len(vectorizer.vocabulary_):,} tokens",
        f"  TF-IDF matrix (train)  : {X_train.shape}",
        f"  TF-IDF matrix (test)   : {X_test.shape}",
        f"  combined_text avg words: {int(df['combined_text'].str.split().str.len().mean()):,}",
        "",
        "  Output files:",
        f"    {STAGE2_PICKLE}",
        f"    {X_TRAIN_PATH}  {X_TEST_PATH}",
        f"    {Y_TRAIN_PATH}  {Y_TEST_PATH}",
        f"    {VECTORIZER_PATH}",
        f"    {LABEL_ENC_PATH}",
        "═" * 60,
    ]
    for line in lines:
        print(line)
        report_lines.append(line)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    report_lines: List[str] = []

    banner = (
        "\n" + "=" * 70 + "\n"
        "  SEC 10-K CLASSIFICATION PROJECT  —  STAGE 2: FEATURE ENGINEERING\n"
        + "=" * 70
    )
    print(banner)
    report_lines.append(banner)

    # ── Load Stage 1 output ───────────────────────────────────────────────────
    if not os.path.exists(STAGE1_PICKLE):
        log.error(
            "Stage 1 output not found at '%s'.\n"
            "  Run stage1_preprocessing.py first.", STAGE1_PICKLE
        )
        sys.exit(1)

    log.info("Loading Stage 1 DataFrame from %s …", STAGE1_PICKLE)
    df = pd.read_pickle(STAGE1_PICKLE)
    log.info("Loaded: %s rows × %s cols", *df.shape)

    # ── STEP 1: Apply risk labels ─────────────────────────────────────────────
    df = apply_risk_labels(df)

    # ── STEP 2: Class distribution ────────────────────────────────────────────
    print_class_distribution(df, report_lines)
    print_sample_scores(df, n=min(6, len(df)))

    # ── STEP 3: Build combined_text ───────────────────────────────────────────
    df = build_combined_text(df)

    # ── STEP 4: TF-IDF vectorisation ─────────────────────────────────────────
    vectorizer, X = build_tfidf_features(
        df["combined_text"],
        max_features = 5000,
        ngram_range  = (1, 2),
    )

    # ── STEP 5: Encode labels & split ─────────────────────────────────────────
    le, y_encoded = encode_labels(df["risk_label"])

    X_train, X_test, y_train, y_test = stratified_split(
        X, y_encoded, test_size=0.20, random_state=42
    )

    # ── Diagnostic: top TF-IDF features per class ────────────────────────────
    print_tfidf_top_features(vectorizer, df, n_top=12)

    # ── STEP 6: Save all artefacts ────────────────────────────────────────────
    save_artefacts(df, X_train, X_test, y_train, y_test, vectorizer, le)

    # ── Final summary ─────────────────────────────────────────────────────────
    print_final_summary(df, X_train, X_test, vectorizer, le, report_lines)

    # ── Write text report ─────────────────────────────────────────────────────
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    log.info("Report saved → %s", REPORT_PATH)

    print("\n✅  Stage 2 complete. Proceed to Stage 3: Model Training.\n")
    return df, X_train, X_test, y_train, y_test, vectorizer, le


if __name__ == "__main__":
    df, X_train, X_test, y_train, y_test, vectorizer, le = main()
