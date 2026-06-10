# ============================================================
# Model Evaluation and Comparison for SEC 10-K Classification
# Models: XGBoost, AdaBoost, CatBoost
# ============================================================

# -------------------------
# Imports
# -------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# -------------------------------------------------------
# Class labels (order should match encoded labels)
# Example:
# 0 = high
# 1 = medium
# 2 = low
# -------------------------------------------------------
class_names = ['high', 'medium', 'low']

# -------------------------------------------------------
# Store models in a dictionary
# -------------------------------------------------------
models = {
    "XGBoost": xgb_model,
    "AdaBoost": ada_model,
    "CatBoost": cat_model
}

# -------------------------------------------------------
# DataFrame to store summary metrics
# -------------------------------------------------------
summary_results = []

# -------------------------------------------------------
# Create subplots for confusion matrices
# -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# -------------------------------------------------------
# Loop through models
# -------------------------------------------------------
for idx, (model_name, model) in enumerate(models.items()):

    # ------------------------------------
    # Predictions
    # ------------------------------------
    y_pred = model.predict(X_test)

    # ------------------------------------
    # Probability predictions for ROC-AUC
    # ------------------------------------
    y_prob = model.predict_proba(X_test)

    # ------------------------------------
    # Metrics
    # ------------------------------------
    accuracy = accuracy_score(y_test, y_pred)

    weighted_precision = precision_score(
        y_test,
        y_pred,
        average='weighted'
    )

    weighted_recall = recall_score(
        y_test,
        y_pred,
        average='weighted'
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average='weighted'
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob,
        multi_class='ovr',
        average='macro'
    )

    # ------------------------------------
    # Classification report (per-class metrics)
    # ------------------------------------
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )

    # ------------------------------------
    # Confusion matrix
    # ------------------------------------
    cm = confusion_matrix(y_test, y_pred)

    # ------------------------------------
    # Print results
    # ------------------------------------
    print("\n" + "=" * 60)
    print(f"{model_name} Results")
    print("=" * 60)

    print(f"Accuracy            : {accuracy:.4f}")
    print(f"Weighted Precision  : {weighted_precision:.4f}")
    print(f"Weighted Recall     : {weighted_recall:.4f}")
    print(f"Weighted F1-score   : {weighted_f1:.4f}")
    print(f"Macro ROC-AUC       : {roc_auc:.4f}")

    print("\nPer-Class Metrics:")
    print(report)

    print("Confusion Matrix:")
    print(cm)

    # ------------------------------------
    # Store summary results
    # ------------------------------------
    summary_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Weighted Precision": weighted_precision,
        "Weighted Recall": weighted_recall,
        "Weighted F1 Score": weighted_f1,
        "Macro ROC-AUC": roc_auc
    })

    # ------------------------------------
    # Plot confusion matrix
    # ------------------------------------
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[idx]
    )

    axes[idx].set_title(f"{model_name}")
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")

# -------------------------------------------------------
# Adjust layout and save figure
# -------------------------------------------------------
plt.tight_layout()

# Save image
plt.savefig("confusion_matrices.png", dpi=300)

# Show plot
plt.show()

# ======================================================
# Create comparison summary table
# ======================================================
summary_df = pd.DataFrame(summary_results)

# Round values for readability
summary_df = summary_df.round(4)

# ------------------------------------------------------
# Print summary table
# ------------------------------------------------------
print("\n")
print("=" * 80)
print("MODEL COMPARISON SUMMARY")
print("=" * 80)

print(summary_df)

# ======================================================
# Determine best model
# (Based primarily on weighted F1-score)
# ======================================================
best_model = summary_df.loc[
    summary_df["Weighted F1 Score"].idxmax(),
    "Model"
]

best_f1 = summary_df["Weighted F1 Score"].max()
best_auc = summary_df.loc[
    summary_df["Weighted F1 Score"].idxmax(),
    "Macro ROC-AUC"
]

# ======================================================
# Print brief justification
# ======================================================
print("\n")
print("=" * 80)
print("MODEL SELECTION JUSTIFICATION")
print("=" * 80)

print(
    f"{best_model} is selected as the best-performing model "
    f"because it achieved the highest weighted F1-score ({best_f1:.4f}), "
    f"which provides a balanced assessment of precision and recall across "
    f"all classes. The model also demonstrated strong discriminative ability "
    f"with a macro ROC-AUC of {best_auc:.4f}. In addition, its confusion "
    f"matrix shows comparatively fewer misclassifications among the high, "
    f"medium, and low classes. Therefore, {best_model} provides the most "
    f"reliable overall performance for the SEC 10-K classification task."
)