import pandas as pd

def filter_consumed_meals(data, out_path):
    print(f"Processing {data.shape[0]} meals (consumed or not)")

    data_filtered = data[(data["Amount Consumed"] == 100.0) | (data["Amount Consumed"] == 1.0)].copy()
    data_deleted = data[~((data["Amount Consumed"] == 100.0) | (data["Amount Consumed"] == 1.0))].copy()

    print(f"After removing not consumed meals: {data_filtered.shape[0]}, removed {data_deleted.shape[0]}")

    data_filtered.to_csv(out_path, index=False)
    return data_filtered


if __name__ == "__main__":
    data_path = "..\splits\lunch_dinner\data_correct_lunch_dinner_visible_gl_stats.csv"
    out_path = '../splits/lunch_dinner/data_all_lunch_dinner_consumed_gl_stats.csv'

    data = pd.read_csv(data_path)

    data_filtered = filter_consumed_meals(data, out_path)