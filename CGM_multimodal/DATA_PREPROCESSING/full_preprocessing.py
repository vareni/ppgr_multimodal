import pandas as pd
from _0_parse_raw_data import parse_raw_data
from _1_detect_wrapped import detect_wrapped
from _2_filter_consumed import filter_consumed_meals
from _3_manual_filter_data import manual_filter_data
from _4_meal_times_filter import meal_times_filter
from _5_add_microbiome import add_microbiome
from _6_split_train_val_test_to_txt import split_train_val_test


if __name__ == "__main__":
    save_folder = "libre_int_5_mins"

    save_all_path = f'../splits/{save_folder}/data_all_lunch_dinner_gl_stats.csv'
    save_correct_path = f'../splits/{save_folder}/data_correct_lunch_dinner_gl_stats.csv'
    save_incorrect_path = f'../splits/{save_folder}/data_incorrect_lunch_dinner_gl_stats.csv'

    OUT_VISIBLE = f"../splits/{save_folder}/data_correct_lunch_dinner_visible_gl_stats.csv"
    OUT_WRAPPED = f"../splits/{save_folder}/data_correct_lunch_dinner_wrapped_gl_stats.csv"

    out_csv = f"../splits/{save_folder}/data_filtered_manually_gl_stats.csv"

    out_path = f'../splits/{save_folder}/data_all_lunch_dinner_consumed_gl_stats.csv'

    out_csv_times = f'../splits/{save_folder}/filter_before_60_after_120.csv'

    train_txt = f"../splits/{save_folder}/train_test_val_60_120_gl_stats/train.txt"  # lunch_dinner/train_test_val_120_120/
    val_txt = f"../splits/{save_folder}/train_test_val_60_120_gl_stats/val.txt"
    test_txt = f"../splits/{save_folder}/train_test_val_60_120_gl_stats/test.txt"

    add_cols = ["CGM_mean_30min_before", "CGM_std_30min_before", "CGM_slope_30min_before",
                "CGM_mean_15min_before", "CGM_std_15min_before", "CGM_slope_15min_before"]


    data_gathered = parse_raw_data(save_all_path, save_correct_path, save_incorrect_path, get_all_meals=False,
                                  meal_types=["lunch", "dinner"])

    visible_df = detect_wrapped(data_gathered, OUT_VISIBLE, OUT_WRAPPED)
    # visible_df = pd.read_csv(OUT_VISIBLE)
    data_consumed = filter_consumed_meals(visible_df, out_path)
    data_filtered_manual = manual_filter_data(data_consumed, out_csv)
    clean_data = meal_times_filter(data_filtered_manual, out_csv)
    clean_data = add_microbiome(clean_data)
    split_train_val_test(clean_data, train_txt, val_txt, test_txt, add_cols)