import pandas as pd


def meal_times_filter(data, out_csv, all_meals_path='../splits/all_meals.csv', after_mins=120, before_mins=60):
    all_meals = pd.read_csv(all_meals_path)
    all_meals["Timestamp"] = pd.to_datetime(all_meals["Timestamp"])
    df = all_meals.sort_values(["sub", "Timestamp"]).copy()

    df["Timestamp_before"] = df.groupby("sub")["Timestamp"].shift(1)
    df["Timestamp_after"] = df.groupby("sub")["Timestamp"].shift(-1)

    df["Minutes_before"] = (
        (df["Timestamp"] - df["Timestamp_before"])
        .dt.total_seconds()
        .div(60)
    )

    df["Minutes_after"] = (
        (df["Timestamp_after"] - df["Timestamp"])
        .dt.total_seconds()
        .div(60)
    )

    data["Timestamp"] = pd.to_datetime(data["Timestamp"])

    cols_to_add = [
        "Timestamp",
        "Timestamp_before",
        "Timestamp_after",
        "Minutes_before",
        "Minutes_after"
    ]

    data = data.merge(
        df[cols_to_add],
        on="Timestamp",
        how="left"
    )

    clean_data = data[
        ((data["Minutes_after"].isna()) | (data["Minutes_after"] >= after_mins)) &
        ((data["Minutes_before"].isna()) | (data["Minutes_before"] >= before_mins))
    ].copy()

    print(f"Meals before times filter: {data.shape[0]}, meals after filter: {clean_data.shape[0]}")

    clean_data.to_csv(out_csv, index=False)

    return clean_data


if __name__ == "__main__":
    out_csv = '../splits/lunch_dinner/filter_before_60_after_120.csv'
    data_path = "../splits/lunch_dinner/data_filtered_manually_gl_stats.csv"

    data = pd.read_csv(data_path)

    clean_data = meal_times_filter(data, out_csv)