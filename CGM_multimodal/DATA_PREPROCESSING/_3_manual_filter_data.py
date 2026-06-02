import pandas as pd

DELETE_LIST = [
    # lunch
    ("Lunch", 10, "00000005-PHOTO-2020-6-21-11-47-0"),

    ("Lunch", 12, "00000029-PHOTO-2023-3-1-14-23-0"),
    ("Lunch", 12, "00000048-PHOTO-2023-3-4-15-1-0"),

    ("Lunch", 14, "00000005-PHOTO-2023-5-9-11-51-0"),
    ("Lunch", 14, "00000028-PHOTO-2023-5-12-12-38-0"),
    ("Lunch", 14, "00000058-PHOTO-2023-5-15-11-42-0"),
    ("Lunch", 14, "00000090-PHOTO-2023-5-18-12-25-0"),

    ("Lunch", 22, "00000052-PHOTO-2021-4-1-12-35-0"),
    ("Lunch", 22, "00000071-PHOTO-2021-4-3-12-29-0"),
    ("Lunch", 22, "00000104-PHOTO-2021-4-6-12-20-0"),

    ("Lunch", 23, "00000047-PHOTO-2021-5-10-13-25-0"),
    ("Lunch", 23, "00000081-PHOTO-2021-5-14-13-26-0"),

    ("Lunch", 28, "00000005-PHOTO-2023-11-28-12-40-0"),
    ("Lunch", 28, "00000036-PHOTO-2023-12-3-12-23-0"),

    ("Lunch", 32, "00000025-PHOTO-2022-1-6-12-47-0"),

    # ("Lunch", 33, "00000017-PHOTO-2022-4-17-19-18-0"), 00000018-PHOTO-2022-4-17-19-18-0 - empty -> 00000017-PHOTO-2022-4-17-19-18-0

    ("Lunch", 34, "00000089-PHOTO-2022-3-7-11-53-0"),
    ("Lunch", 36, "00000010-PHOTO-2022-3-30-12-7-0"),
    ("Lunch", 42, "00000017-PHOTO-2025-7-26-11-27-0"),
    ("Lunch", 44, "00000022-PHOTO-2022-10-18-13-53-0"),
    ("Lunch", 46, "00000017-PHOTO-2025-5-3-11-47-0"),
    ("Lunch", 48, "00000087-PHOTO-2022-11-23-13-22-0"),

    # dinner
    ("Dinner", 4, "00000053-PHOTO-2023-9-15-20-45-0"),
    ("Dinner", 4, "00000064-PHOTO-2023-9-16-22-39-0"),
    ("Dinner", 4, "00000075-PHOTO-2023-9-17-20-2-0"),

    ("Dinner", 10, "00000008-PHOTO-2020-6-21-18-43-0"),

    ("Dinner", 11, "dinner-PHOTO-2020-10-4-19-0-0"),

    ("Dinner", 14, "00000007-PHOTO-2023-5-9-19-1-0"),
    ("Dinner", 14, "00000013-PHOTO-2023-5-10-17-40-0"),
    ("Dinner", 14, "00000070-PHOTO-2023-5-16-20-2-0"),
    ("Dinner", 14, "00000083-PHOTO-2023-5-17-19-46-0"),
    ("Dinner", 14, "00000093-PHOTO-2023-5-18-19-11-0"),

    ("Dinner", 17, "00000014-PHOTO-2023-10-11-22-35-0"),
    ("Dinner", 17, "00000047-PHOTO-2023-10-14-18-47-0"),

    ("Dinner", 26, "00000045-PHOTO-2021-3-31-17-10-0"),

    ("Dinner", 34, "00000103-PHOTO-2022-3-8-17-59-0"),

    ("Dinner", 43, "00000120-PHOTO-2025-10-21-18-4-0"),
    ("Dinner", 43, "00000127-PHOTO-2025-10-21-23-0-0"),
    ("Dinner", 43, "00000157-PHOTO-2025-10-23-22-34-0"),
]


def manual_filter_data(data_filtered, out_csv):
    mask_change = (
        (data_filtered["Meal Type"] == "Lunch") &
        (data_filtered["sub"].astype(int) == 33) &
        (data_filtered["Image path"].astype(str).str.contains(
            "00000018-PHOTO-2022-4-17-19-18-0",
            regex=False
        ))
    )

    data_filtered.loc[mask_change, "Image path"] = (
        data_filtered.loc[mask_change, "Image path"]
        .astype(str)
        .str.replace(
            "00000018-PHOTO-2022-4-17-19-18-0",
            "00000017-PHOTO-2022-4-17-19-18-0",
            regex=False
        )
    )

    mask = False
    not_found = []

    for meal_type, sub, img in DELETE_LIST:
        m = (
            (data_filtered["Meal Type"] == meal_type.lower()) &
            (data_filtered["sub"].astype(int) == sub) &
            (data_filtered["Image path"].astype(str).str.contains(img, regex=False))
        )
        if m.sum() == 0:
            not_found.append((meal_type, sub, img))
        mask = mask | m


    for item in not_found:
        print(f"Not found: {item}")


    print("Rows before:", len(data_filtered))
    print("Rows to delete:", mask.sum())

    data_filtered_clean = data_filtered[~mask].copy()

    print("Rows after:", len(data_filtered_clean))

    data_filtered_clean.to_csv(out_csv, index=False)

    return data_filtered_clean

if __name__ == "__main__":
    csv_path = "../splits/lunch_dinner/data_all_lunch_dinner_consumed_gl_stats.csv"
    out_csv = "../splits/lunch_dinner/data_filtered_manually_gl_stats.csv"

    data_filtered = pd.read_csv(csv_path)

    data_filtered_clean = manual_filter_data(data_filtered, out_csv)