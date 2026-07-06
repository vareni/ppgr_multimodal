import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path

def leave_one_subject_out_pearson(df, subject_col="subject_id", true_col="y_true", pred_col="y_pred"):
    rows = []

    for sid in df[subject_col].unique():
        df_sub = df[df[subject_col] != sid]

        y_true = df_sub[true_col].to_numpy()
        y_pred = df_sub[pred_col].to_numpy()

        r, p = pearsonr(y_true, y_pred)

        rows.append({
            "left_out_subject": sid,
            "n_meals_remaining": len(df_sub),
            "pearson_r": r,
            "p_value": p,
        })

    return pd.DataFrame(rows)
def patient_bootstrap_fisher_ci(
    df,
    subject_col="subject_id",
    true_col="y_true",
    pred_col="y_pred",
    n_boot=10000,
    confidence=0.95,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    subjects = df[subject_col].dropna().unique()
    n_subjects = len(subjects)

    boot_z = []

    for _ in range(n_boot):
        sampled_subjects = rng.choice(subjects, size=n_subjects, replace=True)

        sampled_df = pd.concat(
            [df[df[subject_col] == sid] for sid in sampled_subjects],
            ignore_index=True
        )

        y_true = sampled_df[true_col].to_numpy()
        y_pred = sampled_df[pred_col].to_numpy()

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) < 4:
            continue

        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            continue

        r, _ = pearsonr(y_true, y_pred)

        # Avoid infinities if r is exactly -1 or 1
        r = np.clip(r, -0.999999, 0.999999)

        z = np.arctanh(r)
        boot_z.append(z)

    boot_z = np.array(boot_z)

    alpha = 1 - confidence

    z_low = np.percentile(boot_z, 100 * alpha / 2)
    z_high = np.percentile(boot_z, 100 * (1 - alpha / 2))

    ci_low = np.tanh(z_low)
    ci_high = np.tanh(z_high)

    y_true_full = df[true_col].to_numpy()
    y_pred_full = df[pred_col].to_numpy()

    mask = np.isfinite(y_true_full) & np.isfinite(y_pred_full)
    y_true_full = y_true_full[mask]
    y_pred_full = y_pred_full[mask]

    r_original, p_value = pearsonr(y_true_full, y_pred_full)

    return {
        "pearson_r": r_original,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "n_subjects": n_subjects,
        "n_meals": len(df),
        "n_boot_valid": len(boot_z),
    }

def save_results(ci_result, loo_df, results_dir='results', model_name="CGMHead_MLP_iAUC"):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(results_dir, f"{model_name}_ci_loo_report.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write("Target: iAUC\n")
        f.write("Evaluation split: held-out test subjects\n")
        f.write("CI method: patient-level bootstrap + Fisher z-transform\n")
        f.write("LOO method: leave-one-subject-out on test subjects\n")
        f.write("\n")

        f.write("=== Pearson correlation with confidence interval ===\n")
        f.write(f"Pearson r:     {ci_result['pearson_r']:.6f}\n")
        f.write(f"95% CI low:    {ci_result['ci_low']:.6f}\n")
        f.write(f"95% CI high:   {ci_result['ci_high']:.6f}\n")
        f.write(f"p-value:       {ci_result['p_value']:.6e}\n")
        f.write(f"n subjects:    {ci_result['n_subjects']}\n")
        f.write(f"n meals:       {ci_result['n_meals']}\n")
        f.write(f"n boot valid:  {ci_result['n_boot_valid']}\n")
        f.write("\n")

        f.write("=== Leave-one-subject-out robustness ===\n")
        f.write(f"Mean r:        {loo_df['pearson_r'].mean():.6f}\n")
        f.write(f"Std r:         {loo_df['pearson_r'].std():.6f}\n")
        f.write(f"Min r:         {loo_df['pearson_r'].min():.6f}\n")
        f.write(f"Median r:      {loo_df['pearson_r'].median():.6f}\n")
        f.write(f"Max r:         {loo_df['pearson_r'].max():.6f}\n")
        f.write("\n")

        f.write("LOO per subject:\n")
        for _, row in loo_df.iterrows():
            f.write(
                f"Left out subject {int(row['left_out_subject']):02d}: "
                f"n_meals_remaining = {int(row['n_meals_remaining'])}, "
                f"r = {row['pearson_r']:.6f}, "
                f"p = {row['p_value']:.6e}\n"
            )

    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    df_path = "../splits/lunch/test_for_ci/vlm_lingshu7b_macros_micro.csv"
    df = pd.read_csv(df_path)
    ci_res = patient_bootstrap_fisher_ci(df)
    loo_df = leave_one_subject_out_pearson(df)
    save_results(ci_res, loo_df, model_name="vlm_lingshu7b_macros_micro")
