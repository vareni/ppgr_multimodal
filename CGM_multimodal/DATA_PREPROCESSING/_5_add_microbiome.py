import pandas as pd


def add_microbiome(df, micro_path="../microbiome/microbes_family.csv"):
    micro = pd.read_csv(micro_path).iloc[:, 1:]
    cols = micro.drop(columns="subject", inplace=False).max(numeric_only=True).nlargest(5).index #top 5
    print(f"Top 5 microbiome families: {cols.to_list()}")
    micro = micro[cols.to_list() + ["subject"]]
    df["sub"] = pd.to_numeric(df["sub"], errors="coerce").astype("Int64")
    df = df.merge(micro, how="left", left_on="sub", right_on="subject").drop(columns=["subject"])

    return df


if __name__ == "__main__":
    csv_path = "../splits/lunch_dinner/filter_before_60_after_120.csv"
    df = pd.read_csv(csv_path)

    df = add_microbiome(df)
