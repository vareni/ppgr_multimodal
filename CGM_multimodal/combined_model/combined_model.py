import numpy as np
import torch
import torch.nn as nn
torch.backends.cudnn.enabled = False

from torchvision import transforms
from torch.utils.data import DataLoader
import scipy
import os
import random
from datetime import datetime
import argparse

from mydataset import Nutrition_RGBD
from models.joint_model import assemble_joint_model


parser = argparse.ArgumentParser(description='PyTorch Nutrition5k Training')

now = datetime.now()
parser.add_argument('--run_name', type=str, default=now.strftime("%Y-%m-%d-%H:%M"))
# parser.add_argument('--rgb_rgb', action='store_true', default=False)
# parser.add_argument('--cgmacros', action='store_true', default=False)
# parser.add_argument('--snapme', action='store_true', default=False)
parser.add_argument('--glucose', action='store_true', default=False)
parser.add_argument('--microbiome', action='store_true', default=False)
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: resnet101)')
parser.add_argument('--rgbd', action='store_true',  help='4 channels')
parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
parser.add_argument('--cgm_model', default='CGMHead', type=str, help='choose CGM Head')
parser.add_argument('--train_macro', default=False, action='store_true')
parser.add_argument('--latent_macro_dim', type=int, default=32)

args = parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None, best_score=None, extra=None):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)

def train_model(model, EPOCHS, train_loader, test_loader, criterion, opt, scheduler,
                len_trainset, len_testset, best_path, device):
    train_losses = []
    test_losses = []
    best_loss = np.inf

    # batch_num_train = len(train_loader)
    # batch_num_test = len(test_loader)

    patience = 15
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        tr_loss = 0.0
        for batch, x in enumerate(train_loader):
            # print(f'train batch {batch + 1} / {batch_num_train}')

            inputs = x[0].to(device)
            inputs_rgbd = x[7].to(device)
            # total_calories = x[2].to(device).float()
            # total_fat = x[4].to(device).float()
            # total_carb = x[5].to(device).float()
            # total_protein = x[6].to(device).float()
            # true_macros = torch.stack([total_calories, total_fat, total_carb, total_protein], dim=-1)
            tab_feats = torch.stack([t.to(device, dtype=torch.float32) for t in x[9:-3]], dim=1)

            yb = torch.stack([x[-2].to(device), x[-1].to(device)], dim=1).to(device)
            # xb = torch.cat([true_macros, tab_feats], dim=-1)

            # xb, yb = xb.to(device), yb.to(device)
            # pred = model(tab_feats, true_macros=true_macros)
            pred = model(tab_feats, img=inputs, img_depth=inputs_rgbd)
            loss = criterion(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += loss.item() * yb.size(0)
        tr_loss /= len_trainset

        model.eval()
        te_loss = 0.0

        with torch.no_grad():
            for batch, x in enumerate(test_loader):
                # print(f'test batch {batch + 1} / {batch_num_test}')

                inputs = x[0].to(device)
                inputs_rgbd = x[7].to(device)
                # total_calories = x[2].to(device).float()
                # total_fat = x[4].to(device).float()
                # total_carb = x[5].to(device).float()
                # total_protein = x[6].to(device).float()
                # true_macros = torch.stack([total_calories, total_fat, total_carb, total_protein], dim=-1)
                tab_feats = torch.stack([t.to(device, dtype=torch.float32) for t in x[9:-3]], dim=1)

                yb = torch.stack([x[-2].to(device), x[-1].to(device)], dim=1).to(device)
                # xb = torch.cat([true_macros, tab_feats], dim=-1)

                # xb, yb = xb.to(device), yb.to(device)
                # pred = model(tab_feats, true_macros=true_macros)

                pred = model(tab_feats, img=inputs, img_depth=inputs_rgbd)
                loss = criterion(pred, yb)
                te_loss += loss.item() * yb.size(0)
        te_loss /= len_testset

        train_losses.append(tr_loss)
        test_losses.append(te_loss)

        scheduler.step(te_loss)

        if te_loss + 1e-4 < best_loss:
            best_loss = te_loss
            epochs_no_improve = 0
            save_checkpoint(
                best_path,
                model=model,
                optimizer=opt,
                scheduler=scheduler,
                epoch=epoch,
                best_score=best_loss,
            )
            print(f"Saved new best at epoch {epoch}:")

        else:
            epochs_no_improve += 1
        # if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | train loss={tr_loss:.6f} | test loss={te_loss:.6f}")
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break


def eval_model(model, best_path, device, y_std, y_mean, test_loader):
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print("Loaded best model from epoch:", ckpt.get("epoch"), "best_score:", ckpt.get("best_score"))

    pred_list_iauc = []
    gt_list_iauc = []
    pred_list_auc = []
    gt_list_auc = []

    with torch.no_grad():
        for x in test_loader:
            inputs = x[0].to(device)
            inputs_rgbd = x[7].to(device)
            # total_calories = x[2].to(device).float()
            # total_fat = x[4].to(device).float()
            # total_carb = x[5].to(device).float()
            # total_protein = x[6].to(device).float()
            # true_macros = torch.stack([total_calories, total_fat, total_carb, total_protein], dim=-1)
            tab_feats = torch.stack([t.to(device, dtype=torch.float32) for t in x[9:-3]], dim=1)

            yb = torch.stack([x[-2].to(device), x[-1].to(device)], dim=1)
            # xb = torch.cat([true_macros, tab_feats], dim=-1)

            # xb, yb = xb.to(device), yb.to(device) * y_std + y_mean
            # pred = model(xb) * y_std + y_mean
            yb = yb.to(device) * y_std + y_mean
            # pred = model(tab_feats, true_macros=true_macros) * y_std + y_mean
            pred = model(tab_feats, img=inputs, img_depth=inputs_rgbd) * y_std + y_mean

            pred_list_iauc.append(np.exp(pred[:, 0].cpu()))
            gt_list_iauc.append(np.exp(yb[:, 0].cpu()))
            pred_list_auc.append(np.exp(pred[:, 1].cpu()))
            gt_list_auc.append(np.exp(yb[:, 1].cpu()))

    pred_array, gt_array = np.concatenate(pred_list_iauc, axis=0), np.concatenate(gt_list_iauc, axis=0)
    print(f"iAUC: {scipy.stats.pearsonr(pred_array, gt_array)}")

    pred_array, gt_array = np.concatenate(pred_list_auc, axis=0), np.concatenate(gt_list_auc, axis=0)
    print(f"AUC: {scipy.stats.pearsonr(pred_array, gt_array)}")


if __name__ == "__main__":
    SEED = 42
    EPOCHS = 500
    BATCH_SIZE = 16 #64
    g = set_seed(SEED)

    best_path = "CHECKPOINTS/joint_best_rgbd_load.pt"
    best_path_micro = "CHECKPOINTS/joint_best_rgbd_micro_load.pt"

    # TRAIN_TXT = "../cgmacros/lunch_by_subject_combined_kcal_train.txt"
    # TEST_TXT = "../cgmacros/lunch_by_subject_combined_kcal_test.txt"

    # TRAIN_TXT = "../cgmacros/lunch_sub_train_test_val/train.txt"
    # VAL_TXT = "../cgmacros/lunch_sub_train_test_val/val.txt"
    # TEST_TXT = "../cgmacros/lunch_sub_train_test_val/test.txt"

    TRAIN_TXT = "../cgmacros/lunch_dinner_train_val_test/train.txt"
    VAL_TXT = "../cgmacros/lunch_dinner_train_val_test/val.txt"
    TEST_TXT = "../cgmacros/lunch_dinner_train_val_test/test.txt"

    TARGET_COLS = ["iAUC_log", "AUC_log"]
    feature_cols = ['Baseline_Libre', 'Age', 'BMI', 'A1c', 'HOMA', 'Insulin', 'TG', 'Cholesterol',
                    'HDL', 'Non_HDL', 'CHO_HDL_ratio', 'Fasting_BG', 'LDL', 'VLDL',
                    'total_calories', 'total_carb', 'total_protein', 'total_fat',
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

    trainset = Nutrition_RGBD('../cgmacros', TRAIN_TXT, TRAIN_TXT, glucose=True, microbiome=False,
                              transform=train_transform, norm_cols=feature_cols)

    valset = Nutrition_RGBD('../cgmacros', VAL_TXT, VAL_TXT, glucose=True, microbiome=False, transform=test_transform,
                            train=False, stats=trainset.stats, norm_cols=feature_cols)

    testset = Nutrition_RGBD('../cgmacros', TEST_TXT, TEST_TXT, glucose=True, microbiome=False,
                             transform=test_transform, train=False, stats=trainset.stats, norm_cols=feature_cols)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, generator=g, num_workers=4,
                              pin_memory=True) #, persistent_workers=True, prefetch_factor=4)

    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                             pin_memory=True) #, persistent_workers=True, prefetch_factor=4)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}')
    # model = CGMHead(in_dim=len(feature_cols), hidden=64, out_dim=len(TARGET_COLS)).to(device)
    model = assemble_joint_model(args, device)

    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss(beta=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)  # as RGBD
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=15, min_lr=1e-6
    )


    train_model(model, EPOCHS, train_loader, val_loader, criterion, opt, scheduler,
                len_trainset=len(trainset), len_testset=len(valset), best_path=best_path, device=device)

    y_std, y_mean = trainset.stats["std"]["iAUC_log"], trainset.stats["mean"]["iAUC_log"]
    eval_model(model, best_path, device, y_std, y_mean, test_loader)

    # with microbiome

    g = set_seed(SEED)

    trainset = Nutrition_RGBD('../cgmacros', TRAIN_TXT, TRAIN_TXT, glucose=True, microbiome=True,
                              transform=train_transform, norm_cols=feature_cols + feature_cols_micro)

    valset = Nutrition_RGBD('../cgmacros', VAL_TXT, VAL_TXT, glucose=True, microbiome=True, transform=test_transform,
                            train=False, stats=trainset.stats, norm_cols=feature_cols + feature_cols_micro)

    testset = Nutrition_RGBD('../cgmacros', TEST_TXT, TEST_TXT, glucose=True, microbiome=True,
                             transform=test_transform, train=False, stats=trainset.stats,
                             norm_cols=feature_cols + feature_cols_micro)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, generator=g, num_workers=4,
                              pin_memory=True)  # , persistent_workers=True, prefetch_factor=4)

    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                             pin_memory=True)  # , persistent_workers=True, prefetch_factor=4)

    # model = CGMHead(in_dim=len(feature_cols + feature_cols_micro), hidden=64, out_dim=len(TARGET_COLS)).to(device)
    args.microbiome = True
    model = assemble_joint_model(args, device)

    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss(beta=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)  # as RGBD
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=15, min_lr=1e-6
    )


    train_model(model, EPOCHS, train_loader, val_loader, criterion, opt, scheduler,
                len_trainset=len(trainset), len_testset=len(valset), best_path=best_path_micro, device=device)

    y_std, y_mean = trainset.stats["std"]["iAUC_log"], trainset.stats["mean"]["iAUC_log"]
    eval_model(model, best_path_micro, device, y_std, y_mean, test_loader)