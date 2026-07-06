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
from models.joint_model import load_trained_joint_model

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

def eval_model(model, device, y_std, y_mean, test_loader, save_csv):
    pred_list_iauc, gt_list_iauc = [], []
    pred_list_auc, gt_list_auc = [], []
    sub_ids_list = []

    cal_abs, fat_abs, carb_abs, prot_abs = 0., 0., 0., 0.
    cal_gt, fat_gt, carb_gt, prot_gt = 0., 0., 0., 0.
    eps = 1e-8

    with torch.no_grad():
        for x in test_loader:
            inputs = x[0].to(device)
            inputs_rgbd = x[7].to(device)
            tab_feats = torch.stack([t.to(device, dtype=torch.float32) for t in x[9:-4]], dim=1)
            yb = torch.stack([x[-3].to(device), x[-2].to(device)], dim=1)

            total_calories = x[2].to(device).float()
            total_fat = x[4].to(device).float()
            total_carb = x[5].to(device).float()
            total_protein = x[6].to(device).float()

            pred, pred_macros = model(tab_feats, img=inputs, img_depth=inputs_rgbd)

            yb_denorm = yb.to(device) * y_std + y_mean
            pred_denorm = pred * y_std + y_mean

            pred_list_iauc.append(np.exp(pred_denorm[:, 0].cpu()))
            gt_list_iauc.append(np.exp(yb_denorm[:, 0].cpu()))
            pred_list_auc.append(np.exp(pred_denorm[:, 1].cpu()))
            gt_list_auc.append(np.exp(yb_denorm[:, 1].cpu()))
            sub_ids_list.extend(list(x[-1]))

            if pred_macros is not None:
                cal_abs += torch.abs(pred_macros[:, 0] - total_calories).sum().item()
                fat_abs += torch.abs(pred_macros[:, 1] - total_fat).sum().item()
                carb_abs += torch.abs(pred_macros[:, 2] - total_carb).sum().item()
                prot_abs += torch.abs(pred_macros[:, 3] - total_protein).sum().item()
                cal_gt += total_calories.sum().item()
                fat_gt += total_fat.sum().item()
                carb_gt += total_carb.sum().item()
                prot_gt += total_protein.sum().item()

    pred_array, gt_array = np.concatenate(pred_list_iauc), np.concatenate(gt_list_iauc)
    print(f"iAUC: {scipy.stats.pearsonr(pred_array, gt_array)}")
    df_test_pred = pd.DataFrame({
        "subject_id": sub_ids_list,
        "y_true": gt_array,
        "y_pred": pred_array,
    })

    df_test_pred.to_csv(save_csv, index=False)

    pred_array, gt_array = np.concatenate(pred_list_auc), np.concatenate(gt_list_auc)
    print(f"AUC:  {scipy.stats.pearsonr(pred_array, gt_array)}")

    if cal_gt > 0:
        print(f"Macro PMAE — Cal: {100 * cal_abs / (cal_gt + eps):.2f}%  "
              f"Fat: {100 * fat_abs / (fat_gt + eps):.2f}%  "
              f"Carb: {100 * carb_abs / (carb_gt + eps):.2f}%  "
              f"Prot: {100 * prot_abs / (prot_gt + eps):.2f}%")


if __name__ == "__main__":
    chk = 'CHECKPOINTS/joint_mlp_micro_070_49534.pt' #joint_film_micro_074_49533
    model = load_trained_joint_model(chk, args, device=None, out_dim=2)
    save_csv = 'CHECKPOINTS/mlp_results_micro_070.csv'

    SEED = 42
    EPOCHS = 500
    BATCH_SIZE = 16  # 64
    g = set_seed(SEED)

    TRAIN_TXT = "../cgmacros/lunch_sub_train_test_val/train.txt"
    # VAL_TXT = "../cgmacros/lunch_sub_train_test_val/val.txt"
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

    train_transform = transforms.Compose([
        transforms.Resize((320, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((320, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainset = Nutrition_RGBD('../cgmacros', TRAIN_TXT, TRAIN_TXT, glucose=True, microbiome=args.microbiome,
                              transform=train_transform, norm_cols=feature_cols + feature_cols_micro) #feature_cols_micro

    testset = Nutrition_RGBD('../cgmacros', TEST_TXT, TEST_TXT, glucose=True, microbiome=args.microbiome,
                             transform=test_transform, train=False, stats=trainset.stats, norm_cols=feature_cols + feature_cols_micro)

    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                             pin_memory=True)  # , persistent_workers=True, prefetch_factor=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}')

    y_std, y_mean = trainset.stats["std"]["iAUC_log"], trainset.stats["mean"]["iAUC_log"]
    eval_model(model, device, y_std, y_mean, test_loader, save_csv)
