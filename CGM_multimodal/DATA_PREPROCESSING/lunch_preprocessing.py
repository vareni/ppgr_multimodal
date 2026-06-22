from pathlib import Path

from _0_parse_raw_data import parse_raw_data
from _1_detect_wrapped import detect_wrapped
from _5_add_microbiome import add_microbiome
from _6_split_train_val_test_to_txt import split_train_val_test


if __name__ == "__main__":
    save_folder = "lunch"

    Path(f'../splits/{save_folder}').mkdir(parents=True, exist_ok=True)

    save_all_path = f'../splits/{save_folder}/data_all_lunch.csv'
    save_correct_path = f'../splits/{save_folder}/data_correct_lunch.csv'
    save_incorrect_path = f'../splits/{save_folder}/data_incorrect_lunch.csv'

    OUT_VISIBLE = f"../splits/{save_folder}/data_correct_lunch_dinner_visible.csv"
    OUT_WRAPPED = f"../splits/{save_folder}/data_correct_lunch_dinner_wrapped.csv"

    train_txt = f"../splits/{save_folder}/train.txt"
    val_txt = f"../splits/{save_folder}/val.txt"
    test_txt = f"../splits/{save_folder}/test.txt"


    data_gathered = parse_raw_data(save_all_path, save_correct_path, save_incorrect_path, get_all_meals=False,
                                  meal_types=["lunch"])

    visible_df = detect_wrapped(data_gathered, OUT_VISIBLE, OUT_WRAPPED)
    clean_data = add_microbiome(visible_df)
    split_train_val_test(clean_data, train_txt, val_txt, test_txt)