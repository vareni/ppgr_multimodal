import os
import numpy as np
import torch
import pandas as pd
import scipy
import random
from pathlib import Path
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from mydataset_tmp_2 import Nutrition_RGBD
from models.joint_model import load_macro_models

parser = argparse.ArgumentParser(description='PyTorch Nutrition5k Training')

parser.add_argument('--glucose', action='store_true', default=False)
parser.add_argument('--microbiome', action='store_true', default=False)
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: resnet101)')
parser.add_argument('--rgbd', action='store_true',  help='4 channels')
parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
parser.add_argument('--cgm_model', default='CGMHead', type=str, help='choose CGM Head')
parser.add_argument('--train_macro', default=False, action='store_true')
parser.add_argument('--use_latent_macros', type=bool, default=False)
parser.add_argument('--latent_macro_dim', type=int, default=4)
parser.add_argument('--macro_loss_weight', type=float, default=0.6)

args = parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    return g
def eval_model(models, device, test_loader, source_txt, save_txt):
    """
    Predict calories/fat/carbs/protein and write a txt file with the same
    structure as source_txt, but with true macros replaced by predictions.

    Expected txt columns:
        0: image path
        1: image id
        2: total calories
        3: meal type / label, kept unchanged
        4: total carbohydrates
        5: total fat
        6: total protein
        remaining columns: clinical/microbiome/iAUC/AUC, kept unchanged
    """
    CAL_COL = 2
    CARB_COL = 4
    FAT_COL = 5
    PROT_COL = 6

    net_rgb, net_depth, net_cat = models
    for model in (net_rgb, net_depth, net_cat):
        model.to(device)
        model.eval()

    all_pred_macros = []

    cal_abs, fat_abs, carb_abs, prot_abs = 0., 0., 0., 0.
    cal_gt, fat_gt, carb_gt, prot_gt = 0., 0., 0., 0.
    eps = 1e-8

    with torch.no_grad():
        for x in test_loader:
            inputs = x[0].to(device, non_blocking=True)
            inputs_rgbd = x[7].to(device, non_blocking=True)

            total_calories = x[2].to(device, non_blocking=True).float()
            total_fat = x[4].to(device, non_blocking=True).float()
            total_carb = x[5].to(device, non_blocking=True).float()
            total_protein = x[6].to(device, non_blocking=True).float()

            p2, p3, p4, p5 = net_rgb(inputs)
            d2, d3, d4, d5 = net_depth(inputs_rgbd)
            outputs = net_cat([p2, p3, p4, p5], [d2, d3, d4, d5])

            # outputs[1] is mass, so skip it
            pred_cal = outputs[0].view(-1)
            pred_fat = outputs[2].view(-1)
            pred_carb = outputs[3].view(-1)
            pred_protein = outputs[4].view(-1)

            pred_macros = torch.stack(
                [pred_cal, pred_fat, pred_carb, pred_protein],
                dim=1
            )

            all_pred_macros.append(pred_macros.detach().cpu().numpy())

            cal_abs += torch.abs(pred_cal - total_calories).sum().item()
            fat_abs += torch.abs(pred_fat - total_fat).sum().item()
            carb_abs += torch.abs(pred_carb - total_carb).sum().item()
            prot_abs += torch.abs(pred_protein - total_protein).sum().item()

            cal_gt += total_calories.sum().item()
            fat_gt += total_fat.sum().item()
            carb_gt += total_carb.sum().item()
            prot_gt += total_protein.sum().item()

    pred_macros_np = np.concatenate(all_pred_macros, axis=0)

    df = pd.read_csv(source_txt, sep=r"\s+", header=None)

    if len(df) != len(pred_macros_np):
        raise ValueError(
            f"Number of rows in {source_txt} ({len(df)}) does not match "
            f"number of predictions ({len(pred_macros_np)}). "
            "Make sure the DataLoader uses shuffle=False."
        )

    # Replace only macros. Keep all other columns unchanged.
    df.iloc[:, CAL_COL] = pred_macros_np[:, 0]
    df.iloc[:, FAT_COL] = pred_macros_np[:, 1]
    df.iloc[:, CARB_COL] = pred_macros_np[:, 2]
    df.iloc[:, PROT_COL] = pred_macros_np[:, 3]

    Path(save_txt).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_txt, sep=" ", header=False, index=False, float_format="%.6f")

    print(f"Saved predicted-macro txt to: {save_txt}")

    if cal_gt > 0:
        print(f"Macro PMAE — Cal: {100 * cal_abs / (cal_gt + eps):.2f}%  "
              f"Fat: {100 * fat_abs / (fat_gt + eps):.2f}%  "
              f"Carb: {100 * carb_abs / (carb_gt + eps):.2f}%  "
              f"Prot: {100 * prot_abs / (prot_gt + eps):.2f}%")

if __name__ == "__main__":
    chk = './CHECKPOINTS/regression_nutrition_rgbd_resnet101_2026-01-28-18:59/ckpt_best.pth'
    args.resume = chk
    models = load_macro_models(args)

    SEED = 42
    EPOCHS = 500
    BATCH_SIZE = 16  # 64
    g = set_seed(SEED)

    TRAIN_TXT = "../cgmacros/lunch_sub_train_test_val/train.txt"
    VAL_TXT = "../cgmacros/lunch_sub_train_test_val/val.txt"
    TEST_TXT = "../cgmacros/lunch_sub_train_test_val/test.txt"

    TARGET_COLS = ["iAUC_log", "AUC_log"]
    feature_cols = ['Baseline_Libre', 'Age', 'BMI', 'A1c', 'HOMA', 'Insulin', 'TG', 'Cholesterol',
                    'HDL', 'Non_HDL', 'CHO_HDL_ratio', 'Fasting_BG', 'LDL', 'VLDL',
                    # 'total_calories', 'total_carb', 'total_protein', 'total_fat',
                    'Gender',
                    'Fiber']
    feature_cols_micro = ['Lachnospiraceae', 'Streptococcaceae',
                          'Bacteroidaceae', 'Enterobacteriaceae', 'Prevotellaceae'
                          ]

    test_transform = transforms.Compose([
        transforms.Resize((320, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainset = Nutrition_RGBD('../cgmacros', TRAIN_TXT, TRAIN_TXT, glucose=True, microbiome=True,
                              transform=test_transform, norm_cols=feature_cols + feature_cols_micro)

    valset = Nutrition_RGBD('../cgmacros', VAL_TXT, VAL_TXT, glucose=True, microbiome=True, transform=test_transform,
                            train=False, stats=trainset.stats, norm_cols=feature_cols + feature_cols_micro)

    testset = Nutrition_RGBD('../cgmacros', TEST_TXT, TEST_TXT, glucose=True, microbiome=True,
                             transform=test_transform, train=False, stats=trainset.stats,
                             norm_cols=feature_cols + feature_cols_micro)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, generator=g, num_workers=4,
                              pin_memory=True)  # , persistent_workers=True, prefetch_factor=4)

    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                             pin_memory=True)  # , persistent_workers=True, prefetch_factor=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}')

    eval_model(models, device, train_loader, TRAIN_TXT, save_txt='CHECKPOINTS/pred_macros_train.txt')
    eval_model(models, device, val_loader, VAL_TXT, save_txt='CHECKPOINTS/pred_macros_val.txt')
    eval_model(models, device, test_loader, TEST_TXT, save_txt='CHECKPOINTS/pred_macros_test.txt')
