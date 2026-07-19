from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
EXPECTED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]


def load_creditcard_data(path=RAW_DATA_PATH):
    """Load the Kaggle credit card fraud dataset."""
    path = Path(path)
    print(f"Loading data from: {path}")
    return pd.read_csv(path)


def validate_data(df):
    """Report dataset quality and shape checks without mutating the input data."""
    results = {}

    print("=== Dataset Shape ===")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")
    results["shape"] = df.shape

    print("\n=== Column Dtypes ===")
    dtypes = df.dtypes
    print(dtypes.to_string())
    results["dtypes"] = dtypes

    print("\n=== Missing Values (%) ===")
    missing_pct = df.isna().mean().mul(100).sort_values(ascending=False)
    print(missing_pct.to_string(float_format=lambda value: f"{value:.6f}%"))
    results["missing_pct"] = missing_pct

    print("\n=== Duplicate Rows ===")
    duplicate_rows = int(df.duplicated().sum())
    print(f"Duplicate rows: {duplicate_rows:,}")
    results["duplicate_rows"] = duplicate_rows

    print("\n=== Class Balance ===")
    if "Class" not in df.columns:
        print("Class column is missing.")
        results["class_balance"] = None
    else:
        class_counts = df["Class"].value_counts(dropna=False).sort_index()
        class_pct = df["Class"].value_counts(dropna=False, normalize=True).sort_index().mul(100)
        class_balance = pd.DataFrame({"count": class_counts, "percent": class_pct})
        print(class_balance.to_string(float_format=lambda value: f"{value:.6f}%"))
        results["class_balance"] = class_balance

    print("\n=== Time and Amount Stats ===")
    stats = {}
    for column in ["Time", "Amount"]:
        if column not in df.columns:
            print(f"{column}: column is missing.")
            stats[column] = None
            continue

        column_stats = df[column].agg(["min", "max", "mean"])
        stats[column] = column_stats
        print(
            f"{column}: min={column_stats['min']:.6f}, "
            f"max={column_stats['max']:.6f}, mean={column_stats['mean']:.6f}"
        )

    print("\n=== Amount Negative Value Check ===")
    if "Amount" not in df.columns:
        print("Amount column is missing.")
        negative_amount_count = None
    else:
        negative_amount_count = int((df["Amount"] < 0).sum())
        if negative_amount_count:
            print(f"FLAG: Amount has {negative_amount_count:,} negative values.")
        else:
            print("OK: Amount has no negative values.")
    results["time_amount_stats"] = stats
    results["negative_amount_count"] = negative_amount_count

    print("\n=== Expected Column Check ===")
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    extra_columns = [column for column in df.columns if column not in EXPECTED_COLUMNS]
    print(f"Missing expected columns: {missing_columns if missing_columns else 'None'}")
    print(f"Extra columns: {extra_columns if extra_columns else 'None'}")
    results["missing_columns"] = missing_columns
    results["extra_columns"] = extra_columns

    return results


def print_fraud_ratio(df):
    """Print exact fraud/non-fraud counts and ratios."""
    counts = df["Class"].value_counts().sort_index()
    normal = int(counts.get(0, 0))
    fraud = int(counts.get(1, 0))
    total = normal + fraud
    fraud_pct = fraud / total * 100 if total else 0
    normal_pct = normal / total * 100 if total else 0
    normal_to_fraud = normal / fraud if fraud else float("inf")

    print("=== Exact Fraud / Non-Fraud Ratio ===")
    print(f"Normal transactions (Class=0): {normal:,} ({normal_pct:.8f}%)")
    print(f"Fraud transactions (Class=1): {fraud:,} ({fraud_pct:.8f}%)")
    print(f"Normal:Fraud ratio: {normal_to_fraud:.8f}:1")
    print(f"Fraud rate: {fraud}/{total} = {fraud_pct:.8f}%")

    return {
        "normal": normal,
        "fraud": fraud,
        "total": total,
        "normal_pct": normal_pct,
        "fraud_pct": fraud_pct,
        "normal_to_fraud": normal_to_fraud,
    }


def analyze_duplicates(df):
    """Report exact duplicate rows by class and return examples for inspection."""
    duplicate_mask = df.duplicated()
    duplicate_rows = df.loc[duplicate_mask]
    total_duplicates = int(duplicate_mask.sum())
    duplicates_in_class_0 = int((duplicate_rows["Class"] == 0).sum())
    duplicates_in_class_1 = int((duplicate_rows["Class"] == 1).sum())

    class_0_pct = duplicates_in_class_0 / total_duplicates * 100 if total_duplicates else 0
    class_1_pct = duplicates_in_class_1 / total_duplicates * 100 if total_duplicates else 0

    print("=== Duplicate Analysis ===")
    print(f"Total duplicate rows from df.duplicated(): {total_duplicates:,}")
    print(f"Duplicate rows in Class=0: {duplicates_in_class_0:,} ({class_0_pct:.6f}%)")
    print(f"Duplicate rows in Class=1: {duplicates_in_class_1:,} ({class_1_pct:.6f}%)")

    duplicate_groups = df.loc[df.duplicated(keep=False)].copy()
    example_frames = []

    if duplicate_groups.empty:
        print("No duplicate row pairs found.")
        example_pairs = pd.DataFrame()
    else:
        group_columns = list(df.columns)
        for pair_number, (_, group) in enumerate(
            duplicate_groups.groupby(group_columns, sort=False, dropna=False),
            start=1,
        ):
            if len(group) < 2:
                continue

            pair = group.head(2).copy()
            pair.insert(0, "duplicate_pair", pair_number)
            pair.insert(1, "pair_role", ["original", "duplicate"])
            example_frames.append(pair)

            if len(example_frames) == 5:
                break

        example_pairs = pd.concat(example_frames) if example_frames else pd.DataFrame()
        print("\nExample duplicate row pairs, stacked as original then duplicate:")
        if example_pairs.empty:
            print("No duplicate row pairs available to display.")
        else:
            print(example_pairs.to_string())

    return {
        "total_duplicates": total_duplicates,
        "duplicates_in_class_0": duplicates_in_class_0,
        "duplicates_in_class_1": duplicates_in_class_1,
        "example_pairs": example_pairs,
    }


def remove_duplicates(df):
    """Drop exact duplicate rows, keeping the first occurrence."""
    rows_before = len(df)
    deduplicated_df = df.drop_duplicates(keep="first").copy()
    rows_removed = rows_before - len(deduplicated_df)

    print("=== Duplicate Removal ===")
    print(f"Rows removed: {rows_removed:,}")
    print(f"New shape: {deduplicated_df.shape}")

    return deduplicated_df


def engineer_features(df):
    """Create model-ready fraud features and return metadata for splitting."""
    engineered_df = df.copy()
    engineered_df["hour_of_day"] = (engineered_df["Time"] % 86_400) / 3_600
    engineered_df["hour_sin"] = np.sin(2 * np.pi * engineered_df["hour_of_day"] / 24)
    engineered_df["hour_cos"] = np.cos(2 * np.pi * engineered_df["hour_of_day"] / 24)
    engineered_df = engineered_df.drop(columns=["Time"])

    target_col = "Class"
    feature_columns = [column for column in engineered_df.columns if column != target_col]

    print("=== Feature Engineering ===")
    print(f"Final shape: {engineered_df.shape}")
    print("Final feature list:")
    for feature in feature_columns:
        print(f"- {feature}")

    return engineered_df, feature_columns, target_col


def _split_balance(y):
    fraud_count = int((y == 1).sum())
    total = len(y)
    fraud_pct = fraud_count / total * 100 if total else 0
    return {"rows": total, "fraud": fraud_count, "fraud_pct": fraud_pct}


def _print_split_balance(name, y):
    balance = _split_balance(y)
    print(
        f"{name}: rows={balance['rows']:,}, "
        f"fraud={balance['fraud']:,}, fraud_pct={balance['fraud_pct']:.8f}%"
    )
    return balance


def split_data(df, target_col="Class", test_size=0.15, val_size=0.15, random_state=42):
    """Create stratified train/validation/test splits.

    Uses stratified random splitting because this dataset covers only about two days,
    which is not enough range for a meaningful chronological holdout.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    val_fraction_of_train_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction_of_train_val,
        random_state=random_state,
        stratify=y_train_val,
    )

    print("=== Stratified Train / Validation / Test Split ===")
    _print_split_balance("Train", y_train)
    _print_split_balance("Validation", y_val)
    _print_split_balance("Test", y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_processed_data(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    output_dir="data/processed",
):
    """Save train/validation/test features and targets as CSV files."""
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    datasets = {
        "X_train.csv": X_train,
        "X_val.csv": X_val,
        "X_test.csv": X_test,
        "y_train.csv": y_train.to_frame(name=y_train.name or "Class"),
        "y_val.csv": y_val.to_frame(name=y_val.name or "Class"),
        "y_test.csv": y_test.to_frame(name=y_test.name or "Class"),
    }

    print("=== Saving Processed Data ===")
    saved_paths = {}
    for filename, dataset in datasets.items():
        path = output_path / filename
        dataset.to_csv(path, index=False)
        saved_paths[filename] = path
        print(f"Saved {filename}: {path}")

    return saved_paths


def plot_amount_distribution(df, bins=100):
    """Plot Amount distributions for normal and fraud transactions separately."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for class_value, title, axis in [
        (0, "Amount Distribution: Non-Fraud", axes[0]),
        (1, "Amount Distribution: Fraud", axes[1]),
    ]:
        values = df.loc[df["Class"] == class_value, "Amount"]
        axis.hist(values, bins=bins, alpha=0.85)
        axis.set_title(title)
        axis.set_xlabel("Amount")
        axis.set_ylabel("Transaction Count")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()
    return fig


def plot_time_distribution(df, bins=100):
    """Plot Time distributions for normal and fraud transactions separately."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for class_value, title, axis in [
        (0, "Time Distribution: Non-Fraud", axes[0]),
        (1, "Time Distribution: Fraud", axes[1]),
    ]:
        values = df.loc[df["Class"] == class_value, "Time"]
        axis.hist(values, bins=bins, alpha=0.85)
        axis.set_title(title)
        axis.set_xlabel("Time (seconds from dataset start)")
        axis.set_ylabel("Transaction Count")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()
    return fig


def compute_class_correlations(df):
    """Return V-feature correlations with Class, sorted by absolute strength."""
    v_columns = [column for column in df.columns if column.startswith("V")]
    correlations = df[v_columns + ["Class"]].corr(numeric_only=True)["Class"].drop("Class")
    correlations = correlations.dropna()
    return correlations.reindex(correlations.abs().sort_values(ascending=False).index)


def print_top_correlated_features(df, n=10):
    correlations = compute_class_correlations(df)
    top_correlations = correlations.head(n)
    print(f"=== Top {n} V-Features Correlated With Fraud ===")
    print(top_correlations.to_string(float_format=lambda value: f"{value:.8f}"))
    return top_correlations


def plot_top_correlated_feature_histograms(df, n=5, bins=80):
    """Plot fraud/non-fraud overlaid histograms for the top correlated V-features."""
    top_features = compute_class_correlations(df).head(n)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n))
    if n == 1:
        axes = [axes]

    for axis, feature in zip(axes, top_features.index):
        axis.hist(
            df.loc[df["Class"] == 0, feature],
            bins=bins,
            alpha=0.55,
            density=True,
            label="Non-fraud",
        )
        axis.hist(
            df.loc[df["Class"] == 1, feature],
            bins=bins,
            alpha=0.55,
            density=True,
            label="Fraud",
        )
        axis.set_title(f"{feature} by Class (corr={top_features[feature]:.6f})")
        axis.set_xlabel(feature)
        axis.set_ylabel("Density")
        axis.legend()
        axis.grid(alpha=0.25)

    fig.tight_layout()
    plt.show()
    return fig


def describe_time_range(df):
    """Print whether Time looks like cumulative seconds and how many days it spans."""
    time_min = df["Time"].min()
    time_max = df["Time"].max()
    time_span_seconds = time_max - time_min
    time_span_days = time_span_seconds / 86_400
    is_monotonic = df["Time"].is_monotonic_increasing

    print("=== Time Range Check ===")
    print(f"Time min: {time_min:.6f} seconds")
    print(f"Time max: {time_max:.6f} seconds")
    print(f"Time span: {time_span_seconds:.6f} seconds")
    print(f"Approximate days covered: {time_span_days:.4f}")
    print(f"Time column is monotonic increasing in file order: {is_monotonic}")
    print("Interpretation: Time appears to be seconds elapsed from the first transaction.")

    return {
        "time_min": time_min,
        "time_max": time_max,
        "time_span_seconds": time_span_seconds,
        "time_span_days": time_span_days,
        "is_monotonic": is_monotonic,
    }


def _group_stats(df, column):
    grouped = df.groupby("Class")[column].agg(["count", "min", "max", "mean", "median", "std"])
    return grouped


def _pattern_verdict(df, column):
    grouped = _group_stats(df, column)
    if 0 not in grouped.index or 1 not in grouped.index:
        return {
            "verdict": "not usable",
            "standardized_mean_gap": None,
            "reason": "unable to compare because one class is missing",
        }

    mean_0 = grouped.loc[0, "mean"]
    mean_1 = grouped.loc[1, "mean"]
    std_all = df[column].std()
    standardized_mean_gap = abs(mean_1 - mean_0) / std_all if std_all else 0
    verdict = "usable" if standardized_mean_gap >= 0.25 else "not usable"

    return {
        "verdict": verdict,
        "standardized_mean_gap": standardized_mean_gap,
        "reason": "standardized mean gap threshold >= 0.25",
    }


def _pattern_text(df, column):
    grouped = _group_stats(df, column)
    if 0 not in grouped.index or 1 not in grouped.index:
        return f"{column}: unable to compare because one class is missing."

    mean_0 = grouped.loc[0, "mean"]
    mean_1 = grouped.loc[1, "mean"]
    median_0 = grouped.loc[0, "median"]
    median_1 = grouped.loc[1, "median"]
    pattern = _pattern_verdict(df, column)
    standardized_mean_gap = pattern["standardized_mean_gap"]

    if pattern["verdict"] == "usable":
        verdict = "shows a potentially usable univariate pattern"
    else:
        verdict = "does not show a strong standalone mean-separation pattern"

    return (
        f"{column}: {verdict}. "
        f"normal mean={mean_0:.6f}, fraud mean={mean_1:.6f}, "
        f"normal median={median_0:.6f}, fraud median={median_1:.6f}, "
        f"standardized mean gap={standardized_mean_gap:.6f}."
    )


def print_summary_block(df, validation_results, top_features):
    """Print the final EDA summary requested for the notebook."""
    ratio = print_fraud_ratio(df)
    top_5 = top_features.head(5)
    amount_pattern = _pattern_verdict(df, "Amount")
    time_pattern = _pattern_verdict(df, "Time")
    duplicate_analysis = validation_results.get("duplicate_analysis")

    data_quality_issues = []
    if validation_results["duplicate_rows"]:
        data_quality_issues.append(f"{validation_results['duplicate_rows']:,} duplicate rows")
    if validation_results["negative_amount_count"]:
        data_quality_issues.append(f"{validation_results['negative_amount_count']:,} negative Amount values")
    missing_nonzero = validation_results["missing_pct"][validation_results["missing_pct"] > 0]
    if not missing_nonzero.empty:
        data_quality_issues.append("missing values present")
    if validation_results["missing_columns"]:
        data_quality_issues.append(f"missing expected columns: {validation_results['missing_columns']}")
    if validation_results["extra_columns"]:
        data_quality_issues.append(f"extra columns: {validation_results['extra_columns']}")

    print("\n" + "=" * 72)
    print("EDA SUMMARY")
    print("=" * 72)
    print(f"Total transactions: {ratio['total']:,}")
    print(f"Total fraud: {ratio['fraud']:,}")
    print(f"Fraud percentage: {ratio['fraud_pct']:.8f}%")
    print("\nTop 5 V-features most correlated with fraud:")
    for feature, correlation in top_5.items():
        print(f"- {feature}: {correlation:.8f}")
    print("\nAmount / Time pattern notes:")
    print(f"Amount pattern verdict: {amount_pattern['verdict']}")
    print(f"Time pattern verdict: {time_pattern['verdict']}")
    print(f"- {_pattern_text(df, 'Amount')}")
    print(f"- {_pattern_text(df, 'Time')}")
    print("\nDuplicate analysis:")
    if duplicate_analysis:
        print(f"Duplicate rows in Class=0: {duplicate_analysis['duplicates_in_class_0']:,}")
        print(f"Duplicate rows in Class=1: {duplicate_analysis['duplicates_in_class_1']:,}")
    else:
        print("Duplicate analysis was not run.")
    print("\nData quality issues found:")
    if data_quality_issues:
        for issue in data_quality_issues:
            print(f"- {issue}")
    else:
        print("- None found by these checks.")
    print("=" * 72)

    return {
        "ratio": ratio,
        "top_5_features": top_5,
        "amount_pattern_verdict": amount_pattern,
        "time_pattern_verdict": time_pattern,
        "duplicate_analysis": duplicate_analysis,
        "data_quality_issues": data_quality_issues,
    }


def run_creditcard_eda(path=RAW_DATA_PATH):
    """Run the complete first-pass EDA workflow for the credit card fraud dataset."""
    df = load_creditcard_data(path)

    validation_results = validate_data(df)
    print()
    duplicate_analysis = analyze_duplicates(df)
    validation_results["duplicate_analysis"] = duplicate_analysis
    print()
    ratio = print_fraud_ratio(df)
    print()

    plot_amount_distribution(df)
    plot_time_distribution(df)

    top_10 = print_top_correlated_features(df, n=10)
    plot_top_correlated_feature_histograms(df, n=5)

    time_range = describe_time_range(df)
    summary = print_summary_block(df, validation_results, top_10)

    return {
        "df": df,
        "validation": validation_results,
        "duplicate_analysis": duplicate_analysis,
        "ratio": ratio,
        "top_correlations": top_10,
        "time_range": time_range,
        "summary": summary,
    }


def run_feature_pipeline(path=RAW_DATA_PATH):
    """Run the full data preparation pipeline and save processed split files."""
    raw_df = load_creditcard_data(path)
    rows_before_dedup = len(raw_df)

    deduplicated_df = remove_duplicates(raw_df)
    rows_after_dedup = len(deduplicated_df)
    rows_removed = rows_before_dedup - rows_after_dedup

    engineered_df, feature_columns, target_col = engineer_features(deduplicated_df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(engineered_df, target_col=target_col)
    saved_paths = save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)

    split_summary = {
        "train": _split_balance(y_train),
        "validation": _split_balance(y_val),
        "test": _split_balance(y_test),
    }
    summary = {
        "rows_before_deduplication": rows_before_dedup,
        "rows_after_deduplication": rows_after_dedup,
        "rows_removed": rows_removed,
        "feature_columns": feature_columns,
        "target_column": target_col,
        "split_sizes": split_summary,
        "saved_paths": saved_paths,
    }

    print("=== Feature Pipeline Summary ===")
    print(f"Rows before deduplication: {rows_before_dedup:,}")
    print(f"Rows after deduplication: {rows_after_dedup:,}")
    print(f"Rows removed: {rows_removed:,}")
    for split_name, split_info in split_summary.items():
        print(
            f"{split_name}: rows={split_info['rows']:,}, "
            f"fraud={split_info['fraud']:,}, fraud_pct={split_info['fraud_pct']:.8f}%"
        )

    return X_train, X_val, X_test, y_train, y_val, y_test, summary
