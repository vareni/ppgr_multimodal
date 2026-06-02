import pandas as pd
import numpy as np
from urllib.parse import urlparse
from pathlib import PurePosixPath


def write_to_txt(df, out_txt, add_cols=[]):
    feature_cols_micro = ['Fiber', 'Baseline_Libre', 'Age', 'Gender', 'BMI', 'A1c', 'HOMA', 'Insulin', 'TG',
                          'Cholesterol', 'HDL', 'Non HDL', 'LDL', 'VLDL', 'CHO/HDL ratio', 'Fasting BG',
                          'Lachnospiraceae', 'Streptococcaceae', 'Bacteroidaceae', 'Enterobacteriaceae',
                          'Prevotellaceae', 'iAUC', 'AUC'] + add_cols


    with open(out_txt, "w") as f:
        for _, row in df.iterrows():
            image_path = "CGMacros/" + f"CGMacros-{'{0:03}'.format(row['sub'])}/" + row["Image path"]
            path = urlparse(image_path).path
            image_id = PurePosixPath(path).stem
    
            extras_str = " ".join(f"{row[c]:.6f}" for c in feature_cols_micro)
            # extras_str = extras_str + " " + row['Meal Type']
            # Format: image_path image_id total_calories total_mass total_fat total_carb total_protein
            line = "{path} {img_id} {cal:.6f} {mass:.6f} {fat:.6f} {carb:.6f} {prot:.6f}" \
                   "".format(
                path=image_path,
                img_id=image_id,
                cal=row["Calories"],
                mass=1.0,  # костыль
                fat=row["Fat"],
                carb=row["Carb"],
                prot=row["Protein"],
            ) + f" {extras_str}\n"
            f.write(line)
    
    print(f"Wrote {len(df)} lines to {out_txt}")


def split_train_val_test(df, train_txt, val_txt, test_txt, add_cols=[]):
    # split by sub
    counts = df["sub"].value_counts().sort_index()

    rng = np.random.default_rng(42)   # seed for reproducibility
    subs = counts.index.to_numpy().copy()
    rng.shuffle(subs)

    total_meals = counts.sum()
    target_test = 0.15 * total_meals
    target_val  = 0.15 * total_meals

    test_subs = []
    val_subs = []
    meals_in_test = 0
    meals_in_val = 0

    for s in subs:
        s_int = int(s)
        n_meals = int(counts.loc[s])

        if meals_in_test < target_test:
            test_subs.append(s_int)
            meals_in_test += n_meals

        elif meals_in_val < target_val:
            val_subs.append(s_int)
            meals_in_val += n_meals

    test_subs = set(test_subs)
    val_subs = set(val_subs)
    train_subs = set(map(int, counts.index)) - test_subs - val_subs

    print("Total meals:", int(total_meals))
    print("Train meals:", int(counts.loc[list(train_subs)].sum()), "Subjects:", len(train_subs))
    print("Val meals  :", int(counts.loc[list(val_subs)].sum()),   "Subjects:", len(val_subs))
    print("Test meals :", int(counts.loc[list(test_subs)].sum()),  "Subjects:", len(test_subs))

    df_train = df[df["sub"].isin(train_subs)].copy()
    df_val   = df[df["sub"].isin(val_subs)].copy()
    df_test  = df[df["sub"].isin(test_subs)].copy()

    write_to_txt(df_train, train_txt, add_cols=add_cols)
    write_to_txt(df_val, val_txt, add_cols=add_cols)
    write_to_txt(df_test, test_txt, add_cols=add_cols)


if __name__ == "__main__":
    train_txt = "../splits/lunch_dinner/train_test_val_60_120_gl_stats/train.txt"  # lunch_dinner/train_test_val_120_120/
    val_txt = "../splits/lunch_dinner/train_test_val_60_120_gl_stats/val.txt"
    test_txt = "../splits/lunch_dinner/train_test_val_60_120_gl_stats/test.txt"

    csv_path = "../splits/lunch_dinner/filter_before_60_after_120.csv"

    add_cols = ["CGM_mean_30min_before", "CGM_std_30min_before", "CGM_slope_30min_before",
                    "CGM_mean_15min_before", "CGM_std_15min_before", "CGM_slope_15min_before"]

    df = pd.read_csv(csv_path)
    split_train_val_test(df, train_txt, val_txt, test_txt, add_cols)
