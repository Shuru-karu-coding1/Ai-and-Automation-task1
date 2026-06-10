"""
================================================================================
STAGE 3 — Classification Model Training
SEC 10-K Filing Classification Project
================================================================================
Input   : stage2_output/  (X_train, X_test, y_train, y_test, label_encoder)
Outputs : stage3_output/
            ├── xgboost_model.joblib
            ├── adaboost_model.joblib
            ├── catboost_model.joblib
            ├── predictions.pkl           – dict of all predictions + probas
            └── stage3_training_log.txt   – hyperparameters + timing report

Models trained
──────────────
  1. XGBoostClassifier   — gradient boosted trees, native sparse support
  2. AdaBoostClassifier  — adaptive boosting over shallow decision trees
  3. CatBoostClassifier  — ordered boosting, handles sparse via Pool

NOTE on API versions
─────────────────────
  XGBoost  ≥ 2.0  :  use_label_encoder param removed — omitted here
  sklearn  ≥ 1.2  :  AdaBoost 'algorithm' param removed — SAMME is now default
  CatBoost 1.x    :  Pool(sparse_matrix) supported natively; .toarray() NOT needed

Requirements
────────────
  pip install xgboost catboost scikit-learn joblib
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import time
import logging
import warnings
import textwrap
from typing import Dict, Any, List, Tuple

warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp

# XGBoost  (gradient boosted trees)
import xgboost as xgb

# scikit-learn — AdaBoost + base estimator
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# CatBoost  (ordered boosting)
from catboost import CatBoostClassifier, Pool

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

STAGE2_DIR = "stage2_output"
OUT_DIR    = "stage3_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Input paths ───────────────────────────────────────────────────────────────
X_TRAIN_PATH   = os.path.join(STAGE2_DIR, "X_train.pkl")
X_TEST_PATH    = os.path.join(STAGE2_DIR, "X_test.pkl")
Y_TRAIN_PATH   = os.path.join(STAGE2_DIR, "y_train.pkl")
Y_TEST_PATH    = os.path.join(STAGE2_DIR, "y_test.pkl")
LABEL_ENC_PATH = os.path.join(STAGE2_DIR, "label_encoder.joblib")

# ── Output paths ──────────────────────────────────────────────────────────────
XGB_MODEL_PATH  = os.path.join(OUT_DIR, "xgboost_model.joblib")
ADA_MODEL_PATH  = os.path.join(OUT_DIR, "adaboost_model.joblib")
CAT_MODEL_PATH  = os.path.join(OUT_DIR, "catboost_model.joblib")
PREDS_PATH      = os.path.join(OUT_DIR, "predictions.pkl")
LOG_PATH        = os.path.join(OUT_DIR, "stage3_training_log.txt")

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_stage2_artefacts() -> Tuple:
    """
    Load all Stage 2 artefacts.  Raises a clear error if any file is missing.
    """
    required = [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH, LABEL_ENC_PATH]
    missing  = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing Stage 2 files: {missing}\n"
            "  Run stage2_feature_engineering.py first."
        )

    log.info("Loading Stage 2 artefacts …")
    X_train = joblib.load(X_TRAIN_PATH)
    X_test  = joblib.load(X_TEST_PATH)
    y_train = joblib.load(Y_TRAIN_PATH)
    y_test  = joblib.load(Y_TEST_PATH)
    le      = joblib.load(LABEL_ENC_PATH)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    log.info("X_train : %s  type=%s  sparse=%s",
             X_train.shape, type(X_train).__name__, sp.issparse(X_train))
    log.info("X_test  : %s", X_test.shape)

    train_dist = dict(zip(*np.unique(y_train, return_counts=True)))
    test_dist  = dict(zip(*np.unique(y_test,  return_counts=True)))
    label_map  = {i: c for i, c in enumerate(le.classes_)}

    log.info("y_train dist: %s",
             {label_map.get(k, k): v for k, v in train_dist.items()})
    log.info("y_test  dist: %s",
             {label_map.get(k, k): v for k, v in test_dist.items()})
    log.info("Label mapping (int → class): %s", label_map)

    n_classes = len(np.unique(y_train))
    log.info("Number of classes: %d", n_classes)

    return X_train, X_test, y_train, y_test, le, n_classes


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def train_and_time(
    model_name : str,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    report_rows : List[Dict],
) -> Tuple[Any, np.ndarray, np.ndarray, float]:
    """
    Fit a model, record wall-clock training time, and collect predictions.

    Returns
    -------
    model       : fitted model object
    y_pred      : integer class predictions on X_test
    y_proba     : probability matrix (n_samples × n_classes) on X_test
    elapsed_sec : training wall-clock time in seconds
    """
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Training: {model_name}")
    print(sep)

    # ── Fit ───────────────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t_start

    log.info("%s  →  trained in %.4f seconds", model_name, elapsed)

    # ── Predict ───────────────────────────────────────────────────────────────
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Coerce CatBoost integer array output to plain numpy int64
    y_pred  = np.array(y_pred,  dtype=np.int64)
    y_proba = np.array(y_proba, dtype=np.float64)

    # ── Quick accuracy preview ────────────────────────────────────────────────
    accuracy = float(np.mean(y_pred == y_test))
    print(f"  Training time  : {elapsed:.4f}s")
    print(f"  Test accuracy  : {accuracy:.4f}  ({int(accuracy * len(y_test))}/{len(y_test)} correct)")
    print(f"  y_pred sample  : {y_pred[:min(6, len(y_pred))]}")

    report_rows.append({
        "model"        : model_name,
        "train_sec"    : round(elapsed,  4),
        "test_accuracy": round(accuracy, 4),
        "n_train"      : len(y_train),
        "n_test"       : len(y_test),
    })

    return model, y_pred, y_proba, elapsed


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 1 — XGBoost
# ══════════════════════════════════════════════════════════════════════════════

def build_xgboost(n_classes: int, n_train: int) -> xgb.XGBClassifier:
    """
    Build an XGBoostClassifier with production-quality hyperparameters.

    Hyperparameter rationale
    ────────────────────────
    n_estimators=200        : enough rounds for boosting to converge on text features;
                              diminishing returns beyond ~300 for TF-IDF inputs.
    max_depth=6             : XGBoost default; captures feature interactions without
                              overfitting on our relatively small SEC corpus.
    learning_rate=0.1       : standard starting point; lower = more robust but slower.
    subsample=0.8           : row subsampling per tree — reduces variance.
    colsample_bytree=0.8    : feature subsampling per tree — important for sparse
                              TF-IDF where most features are zero.
    eval_metric='mlogloss'  : multi-class log loss — proper scoring for probability
                              calibration across 3 risk classes.
    objective='multi:softmax': direct multi-class without OVR wrappers.
    num_class               : required by XGBoost for multi-class softmax.
    tree_method='hist'      : histogram-based algorithm, much faster on sparse data
                              than the exact method, equivalent accuracy.
    use_label_encoder       : REMOVED in XGBoost ≥ 2.0 — do NOT pass it.
    verbosity=1             : 0=silent, 1=warning, 2=info, 3=debug.

    NOTE: XGBoost natively accepts scipy sparse CSR matrices — no .toarray() needed.
    """
    return xgb.XGBClassifier(
        n_estimators      = 200,
        max_depth         = 6,
        learning_rate     = 0.1,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        eval_metric       = "mlogloss",
        objective         = "multi:softmax",
        num_class         = n_classes,
        tree_method       = "hist",        # fast histogram method for sparse TF-IDF
        random_state      = 42,
        verbosity         = 0,             # suppress XGBoost internal logs
        n_jobs            = -1,            # use all CPU cores
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 2 — AdaBoost
# ══════════════════════════════════════════════════════════════════════════════

def build_adaboost() -> AdaBoostClassifier:
    """
    Build an AdaBoostClassifier with a Decision Tree base estimator.

    Hyperparameter rationale
    ────────────────────────
    estimator=DecisionTreeClassifier(max_depth=3)
        Base learner depth of 3 creates "weak" stumps — shallow enough that
        boosting has room to correct mistakes, but deep enough to capture
        bigram feature interactions (e.g. 'material weakness' vs 'material').

    n_estimators=200        : 200 sequential boosting rounds. AdaBoost converges
                              more slowly than gradient boosting, so 200 is the
                              minimum for stable performance on text features.

    learning_rate=0.5       : Shrinks each estimator's contribution, reducing
                              overfitting. Works synergistically with n_estimators.

    algorithm               : REMOVED in sklearn ≥ 1.2. SAMME.R (real-valued
                              boosting) is now the only supported algorithm.
                              Do NOT pass 'algorithm' keyword.

    IMPORTANT: AdaBoost does NOT natively support sparse matrices — X_train must
    be converted to a dense array via .toarray() before fitting.
    """
    base_estimator = DecisionTreeClassifier(
        max_depth         = 3,
        min_samples_split = 2,
        min_samples_leaf  = 1,
        random_state      = 42,
    )
    return AdaBoostClassifier(
        estimator     = base_estimator,
        n_estimators  = 200,
        learning_rate = 0.5,
        random_state  = 42,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 3 — CatBoost
# ══════════════════════════════════════════════════════════════════════════════

def build_catboost(n_classes: int) -> CatBoostClassifier:
    """
    Build a CatBoostClassifier with sparse-aware configuration.

    Hyperparameter rationale
    ────────────────────────
    iterations=200          : equivalent to n_estimators; 200 trees for text.
    depth=6                 : CatBoost default; symmetric (oblivious) trees at
                              depth 6 give good bias-variance trade-off.
    learning_rate=0.1       : CatBoost auto-tunes this if not set, but explicit
                              value ensures reproducibility.
    loss_function           : 'MultiClass' for 3-label classification.
    eval_metric             : 'Accuracy' shown during training (if verbose > 0).
    l2_leaf_reg=3.0         : L2 regularisation on leaf values — CatBoost default,
                              prevents overfitting on small corpora.
    bootstrap_type='Bernoulli': row subsampling, compatible with sparse input.
    subsample=0.8           : fraction of rows per tree (requires Bernoulli).
    sparse_features_conflict_fraction=0.0
                            : CatBoost internal param for sparse TF-IDF stability.
    verbose=0               : fully silent.
    random_seed=42          : reproducible results.

    SPARSE HANDLING:
        CatBoost Pool(sparse_matrix) works natively since v0.18.
        Two correct options:
          a) model.fit(X_csr, y)              — direct sparse fit ✓
          b) pool = Pool(X_csr, label=y)      — Pool wrapper   ✓
        We use option (b) — Pool gives CatBoost more metadata for optimisation.
    """
    return CatBoostClassifier(
        iterations      = 200,
        depth           = 6,
        learning_rate   = 0.1,
        loss_function   = "MultiClass",
        eval_metric     = "Accuracy",
        l2_leaf_reg     = 3.0,
        bootstrap_type  = "Bernoulli",
        subsample       = 0.8,
        verbose         = 0,
        random_seed     = 42,
        thread_count    = -1,           # use all CPU cores
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SPARSE CONVERSION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def to_dense(X) -> np.ndarray:
    """Convert sparse matrix to dense numpy array (needed for AdaBoost)."""
    if sp.issparse(X):
        return X.toarray()
    return np.array(X)


def to_catboost_pool(X, y) -> Pool:
    """
    Wrap data in a CatBoost Pool object.
    Pool handles sparse CSR natively; passing sparse directly is more memory-
    efficient than converting to dense for large TF-IDF matrices.
    """
    if sp.issparse(X):
        # Ensure CSR format — Pool is optimised for row-access patterns
        X_csr = X.tocsr()
    else:
        X_csr = X
    return Pool(data=X_csr, label=y.astype(int))


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE MODELS
# ══════════════════════════════════════════════════════════════════════════════

def save_models(
    xgb_model : xgb.XGBClassifier,
    ada_model : AdaBoostClassifier,
    cat_model : CatBoostClassifier,
) -> None:
    """
    Persist all three models.

    All saved as .joblib for consistency in Stage 5 API loading.
    CatBoost also supports its native .cbm format, but .joblib keeps
    the loading API uniform across all three models.
    """
    joblib.dump(xgb_model, XGB_MODEL_PATH)
    log.info("Saved XGBoost   → %s  (%.1f KB)",
             XGB_MODEL_PATH, os.path.getsize(XGB_MODEL_PATH) / 1024)

    joblib.dump(ada_model, ADA_MODEL_PATH)
    log.info("Saved AdaBoost  → %s  (%.1f KB)",
             ADA_MODEL_PATH, os.path.getsize(ADA_MODEL_PATH) / 1024)

    joblib.dump(cat_model, CAT_MODEL_PATH)
    log.info("Saved CatBoost  → %s  (%.1f KB)",
             CAT_MODEL_PATH, os.path.getsize(CAT_MODEL_PATH) / 1024)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_training_summary(
    report_rows : List[Dict],
    label_map   : Dict[int, str],
    predictions : Dict,
    log_lines   : List[str],
) -> None:
    """Print a comprehensive training summary table and write to log file."""

    lines = [
        "",
        "═" * 65,
        "  STAGE 3 — TRAINING SUMMARY",
        "═" * 65,
        "",
        f"  {'Model':<20} {'Train (s)':>10} {'Test Acc':>10} {'Train N':>8} {'Test N':>7}",
        "  " + "─" * 58,
    ]

    for row in report_rows:
        lines.append(
            f"  {row['model']:<20} "
            f"{row['train_sec']:>10.4f} "
            f"{row['test_accuracy']:>10.4f} "
            f"{row['n_train']:>8} "
            f"{row['n_test']:>7}"
        )

    lines += ["", "  Label mapping (encoded → class name):"]
    for k, v in sorted(label_map.items()):
        lines.append(f"    {k}  →  {v}")

    lines += ["", "  Prediction snapshots (first 6 test samples):"]
    for model_name, preds in predictions.items():
        if model_name in ("y_test", "label_map"):   # skip metadata keys
            continue
        decoded = [label_map.get(int(p), str(p)) for p in preds["y_pred"][:6]]
        lines.append(f"    {model_name:<20}: {decoded}")

    lines += [
        "",
        "  Saved models:",
        f"    {XGB_MODEL_PATH}",
        f"    {ADA_MODEL_PATH}",
        f"    {CAT_MODEL_PATH}",
        "",
        "  Next step: run stage4_evaluation.py",
        "═" * 65,
    ]

    for line in lines:
        print(line)
        log_lines.append(line)


def write_hyperparameter_log(log_lines: List[str]) -> None:
    """Append hyperparameter documentation to the log."""
    hparam_block = textwrap.dedent("""

    ════════════════════════════════════════════════════════════════════
      HYPERPARAMETER REFERENCE
    ════════════════════════════════════════════════════════════════════

      MODEL 1 — XGBoost
      ─────────────────
      n_estimators=200, max_depth=6, learning_rate=0.1
      subsample=0.8, colsample_bytree=0.8
      eval_metric='mlogloss', objective='multi:softmax'
      tree_method='hist'  [fast sparse-aware histogram splits]
      NOTE: use_label_encoder REMOVED in XGBoost >= 2.0

      MODEL 2 — AdaBoost
      ──────────────────
      base=DecisionTreeClassifier(max_depth=3)
      n_estimators=200, learning_rate=0.5
      NOTE: algorithm param REMOVED in sklearn >= 1.2 (SAMME only)
      NOTE: requires dense input — X.toarray() applied before fit

      MODEL 3 — CatBoost
      ──────────────────
      iterations=200, depth=6, learning_rate=0.1
      loss_function='MultiClass', l2_leaf_reg=3.0
      bootstrap_type='Bernoulli', subsample=0.8
      NOTE: accepts sparse CSR via Pool(X_csr, label=y) natively
    """)
    log_lines.append(hparam_block)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():

    log_lines: List[str]  = []
    report_rows: List[Dict] = []

    banner = (
        "\n" + "=" * 70 + "\n"
        "  SEC 10-K CLASSIFICATION PROJECT  —  STAGE 3: MODEL TRAINING\n"
        + "=" * 70
    )
    print(banner)
    log_lines.append(banner)

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, le, n_classes = load_stage2_artefacts()
    label_map = {int(i): str(c) for i, c in enumerate(le.classes_)}

    # Dense versions (AdaBoost requirement)
    X_train_dense = to_dense(X_train)
    X_test_dense  = to_dense(X_test)

    # CatBoost Pool objects (native sparse, more memory-efficient)
    pool_train = to_catboost_pool(X_train, y_train)
    pool_test  = to_catboost_pool(X_test,  y_test)

    print(f"\n  Data ready:")
    print(f"    X_train  : {X_train.shape}  (sparse={sp.issparse(X_train)})")
    print(f"    X_test   : {X_test.shape}")
    print(f"    n_classes: {n_classes}  →  {label_map}")

    # ── Predictions store ─────────────────────────────────────────────────────
    # Structure:  { model_name: { "y_pred": np.array, "y_proba": np.array } }
    predictions: Dict[str, Dict] = {}

    # ─────────────────────────────────────────────────────────────────────────
    #  MODEL 1 — XGBoost
    #  Accepts sparse CSR natively via tree_method='hist'
    # ─────────────────────────────────────────────────────────────────────────
    xgb_clf = build_xgboost(n_classes, len(y_train))

    xgb_clf, xgb_pred, xgb_proba, xgb_time = train_and_time(
        model_name  = "XGBoost",
        model       = xgb_clf,
        X_train     = X_train,          # sparse OK for XGBoost
        y_train     = y_train,
        X_test      = X_test,
        y_test      = y_test,
        report_rows = report_rows,
    )
    predictions["XGBoost"] = {"y_pred": xgb_pred, "y_proba": xgb_proba}

    # ─────────────────────────────────────────────────────────────────────────
    #  MODEL 2 — AdaBoost
    #  REQUIRES dense input — sparse matrices not supported by sklearn's
    #  AdaBoostClassifier as of v1.x.  Convert with .toarray() first.
    # ─────────────────────────────────────────────────────────────────────────
    ada_clf = build_adaboost()

    print("\n  [AdaBoost] Converting sparse TF-IDF → dense array (required) …")
    log.info("AdaBoost requires dense input — converting X_train/X_test to dense arrays.")

    ada_clf, ada_pred, ada_proba, ada_time = train_and_time(
        model_name  = "AdaBoost",
        model       = ada_clf,
        X_train     = X_train_dense,    # dense: AdaBoost requirement
        y_train     = y_train,
        X_test      = X_test_dense,
        y_test      = y_test,
        report_rows = report_rows,
    )
    predictions["AdaBoost"] = {"y_pred": ada_pred, "y_proba": ada_proba}

    # ─────────────────────────────────────────────────────────────────────────
    #  MODEL 3 — CatBoost
    #  Uses Pool(sparse_matrix) for native sparse support — no .toarray() needed.
    #  Pool is the idiomatic CatBoost data container; it handles sparse CSR,
    #  feature names, and sample weights in one object.
    # ─────────────────────────────────────────────────────────────────────────
    cat_clf = build_catboost(n_classes)

    print("\n  [CatBoost] Using Pool(sparse_matrix) — no dense conversion needed.")
    log.info("CatBoost Pool constructed from sparse CSR matrix.")

    # CatBoost has a special fit signature when using Pool objects
    print(f"\n{'─' * 60}")
    print(f"  Training: CatBoost")
    print(f"{'─' * 60}")

    t_start = time.perf_counter()
    cat_clf.fit(pool_train)              # fit on Pool for best performance
    cat_time = time.perf_counter() - t_start

    cat_pred  = np.array(cat_clf.predict(pool_test),       dtype=np.int64).flatten()
    cat_proba = np.array(cat_clf.predict_proba(pool_test), dtype=np.float64)

    cat_acc = float(np.mean(cat_pred == y_test))
    print(f"  Training time  : {cat_time:.4f}s")
    print(f"  Test accuracy  : {cat_acc:.4f}  ({int(cat_acc * len(y_test))}/{len(y_test)} correct)")
    print(f"  y_pred sample  : {cat_pred[:min(6, len(cat_pred))]}")

    log.info("CatBoost → trained in %.4f seconds", cat_time)
    report_rows.append({
        "model"        : "CatBoost",
        "train_sec"    : round(cat_time, 4),
        "test_accuracy": round(cat_acc,  4),
        "n_train"      : len(y_train),
        "n_test"       : len(y_test),
    })
    predictions["CatBoost"] = {"y_pred": cat_pred, "y_proba": cat_proba}

    # ── Save all models ───────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Saving models …")
    save_models(xgb_clf, ada_clf, cat_clf)

    # ── Save all predictions (used by Stage 4 evaluation) ────────────────────
    predictions["y_test"]    = y_test
    predictions["label_map"] = label_map
    joblib.dump(predictions, PREDS_PATH)
    log.info("Saved predictions → %s", PREDS_PATH)

    # ── Print & write summary ─────────────────────────────────────────────────
    print_training_summary(report_rows, label_map, predictions, log_lines)
    write_hyperparameter_log(log_lines)

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines))
    log.info("Training log saved → %s", LOG_PATH)

    print(f"\n✅  Stage 3 complete. Proceed to Stage 4: Model Evaluation.\n")

    return {
        "xgb_model" : xgb_clf,
        "ada_model" : ada_clf,
        "cat_model" : cat_clf,
        "predictions": predictions,
        "y_test"    : y_test,
        "label_map" : label_map,
    }


if __name__ == "__main__":
    results = main()
