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
parser.add_argument('--latent_macro_dim', type=int, default=8)
parser.add_argument('--use_latent_macros', type=bool, default=False)
parser.add_argument('--macro_loss_weight', type=float, default=0.5)

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


def train_model(model, EPOCHS, train_loader, test_loader, criterion, opt, sched_macro, sched_cgm, #scheduler,
                len_trainset, len_testset, best_path, device, macro_weight=0.5):
    best_loss = np.inf
    patience = 15
    epochs_no_improve = 0
    eps = 1e-8

    for epoch in range(EPOCHS):
        model.train()
        tr_loss = 0.0
        tr_cal_loss = tr_fat_loss = tr_carb_loss = tr_prot_loss = 0.0  # <-- add

        for batch, x in enumerate(train_loader):
            inputs      = x[0].to(device)
            inputs_rgbd = x[7].to(device)
            tab_feats   = torch.stack([t.to(device, dtype=torch.float32) for t in x[9:-3]], dim=1)
            yb          = torch.stack([x[-2].to(device), x[-1].to(device)], dim=1).to(device)

            total_calories = x[2].to(device).float()  # <-- add
            total_fat      = x[4].to(device).float()  # <-- add
            total_carb     = x[5].to(device).float()  # <-- add
            total_protein  = x[6].to(device).float()  # <-- add

            pred, pred_macros = model(tab_feats, img=inputs, img_depth=inputs_rgbd)
            cgm_loss = criterion(pred, yb)

            loss = cgm_loss
            if pred_macros is not None:
                B = total_calories.size(0)
                cal_l  = B * nn.functional.l1_loss(pred_macros[:, 0], total_calories) / total_calories.sum().clamp_min(eps)
                fat_l  = B * nn.functional.l1_loss(pred_macros[:, 1], total_fat)      / total_fat.sum().clamp_min(eps)
                carb_l = B * nn.functional.l1_loss(pred_macros[:, 2], total_carb)     / total_carb.sum().clamp_min(eps)
                prot_l = B * nn.functional.l1_loss(pred_macros[:, 3], total_protein)  / total_protein.sum().clamp_min(eps)
                macro_loss = cal_l + fat_l + carb_l + prot_l
                loss = cgm_loss + macro_weight * macro_loss

                tr_cal_loss  += cal_l.item()   # <-- add
                tr_fat_loss  += fat_l.item()   # <-- add
                tr_carb_loss += carb_l.item()  # <-- add
                tr_prot_loss += prot_l.item()  # <-- add

            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * yb.size(0)

        tr_loss /= len_trainset
        n_batches = len(train_loader)

        # --- val loop (same pattern) ---
        model.eval()
        te_loss = 0.0
        te_cal_loss = te_fat_loss = te_carb_loss = te_prot_loss = 0.0  # <-- add

        with torch.no_grad():
            for batch, x in enumerate(test_loader):
                inputs      = x[0].to(device)
                inputs_rgbd = x[7].to(device)
                tab_feats   = torch.stack([t.to(device, dtype=torch.float32) for t in x[9:-3]], dim=1)
                yb          = torch.stack([x[-2].to(device), x[-1].to(device)], dim=1).to(device)

                total_calories = x[2].to(device).float()  # <-- add
                total_fat      = x[4].to(device).float()  # <-- add
                total_carb     = x[5].to(device).float()  # <-- add
                total_protein  = x[6].to(device).float()  # <-- add

                pred, pred_macros = model(tab_feats, img=inputs, img_depth=inputs_rgbd)
                cgm_loss = criterion(pred, yb)

                loss = cgm_loss
                if pred_macros is not None:
                    B = total_calories.size(0)
                    cal_l  = B * nn.functional.l1_loss(pred_macros[:, 0], total_calories) / total_calories.sum().clamp_min(eps)
                    fat_l  = B * nn.functional.l1_loss(pred_macros[:, 1], total_fat)      / total_fat.sum().clamp_min(eps)
                    carb_l = B * nn.functional.l1_loss(pred_macros[:, 2], total_carb)     / total_carb.sum().clamp_min(eps)
                    prot_l = B * nn.functional.l1_loss(pred_macros[:, 3], total_protein)  / total_protein.sum().clamp_min(eps)
                    loss = cgm_loss + macro_weight * (cal_l + fat_l + carb_l + prot_l)

                    te_cal_loss  += cal_l.item()   # <-- add
                    te_fat_loss  += fat_l.item()   # <-- add
                    te_carb_loss += carb_l.item()  # <-- add
                    te_prot_loss += prot_l.item()  # <-- add

                te_loss += loss.item() * yb.size(0)
        te_loss /= len_testset
        n_val_batches = len(test_loader)

        # scheduler.step(te_loss)
        sched_macro.step()  # ExponentialLR needs no argument
        sched_cgm.step(te_loss)  # ReduceLROnPlateau needs the metric

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={tr_loss:.6f} | test loss={te_loss:.6f}\n"
            f"  train macros — cal: {tr_cal_loss/n_batches:.4f}  fat: {tr_fat_loss/n_batches:.4f}  "
            f"carb: {tr_carb_loss/n_batches:.4f}  prot: {tr_prot_loss/n_batches:.4f}\n"
            f"  val   macros — cal: {te_cal_loss/n_val_batches:.4f}  fat: {te_fat_loss/n_val_batches:.4f}  "
            f"carb: {te_carb_loss/n_val_batches:.4f}  prot: {te_prot_loss/n_val_batches:.4f}"
        )

        if te_loss + 1e-4 < best_loss:
            best_loss = te_loss
            epochs_no_improve = 0
            save_checkpoint(best_path, model=model, optimizer=opt, scheduler=None, # scheduler,
                            epoch=epoch, best_score=best_loss)
            print(f"  => Saved new best")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break


def eval_model(model, best_path, device, y_std, y_mean, test_loader):
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    pred_list_iauc, gt_list_iauc = [], []
    pred_list_auc, gt_list_auc = [], []

    cal_abs, fat_abs, carb_abs, prot_abs = 0., 0., 0., 0.
    cal_gt, fat_gt, carb_gt, prot_gt = 0., 0., 0., 0.
    eps = 1e-8

    with torch.no_grad():
        for x in test_loader:
            inputs = x[0].to(device)
            inputs_rgbd = x[7].to(device)
            tab_feats = torch.stack([t.to(device, dtype=torch.float32) for t in x[9:-3]], dim=1)
            yb = torch.stack([x[-2].to(device), x[-1].to(device)], dim=1)

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
    pred_array, gt_array = np.concatenate(pred_list_auc), np.concatenate(gt_list_auc)
    print(f"AUC:  {scipy.stats.pearsonr(pred_array, gt_array)}")

    if cal_gt > 0:
        print(f"Macro PMAE — Cal: {100 * cal_abs / (cal_gt + eps):.2f}%  "
              f"Fat: {100 * fat_abs / (fat_gt + eps):.2f}%  "
              f"Carb: {100 * carb_abs / (carb_gt + eps):.2f}%  "
              f"Prot: {100 * prot_abs / (prot_gt + eps):.2f}%")

def weighted_cgm_loss(pred, target, auc_weight=0.9):
    loss_iauc = nn.functional.smooth_l1_loss(pred[:, 0], target[:, 0], beta=1.0)
    loss_auc  = nn.functional.smooth_l1_loss(pred[:, 1], target[:, 1], beta=1.0)
    return loss_iauc + auc_weight * loss_auc

class PartialLRScheduler:
    """Wraps a scheduler to only touch param groups at given indices."""
    def __init__(self, scheduler, group_indices: list[int]):
        self.scheduler = scheduler
        self.group_indices = group_indices
        # stash lrs of groups we don't want touched
        self._other_lrs = None

    def step(self, *args, **kwargs):
        opt = self.scheduler.optimizer
        # freeze lrs of groups we don't own
        saved = {i: pg["lr"] for i, pg in enumerate(opt.param_groups)
                 if i not in self.group_indices}
        self.scheduler.step(*args, **kwargs)
        # restore them
        for i, lr in saved.items():
            opt.param_groups[i]["lr"] = lr

def define_opt_and_schedulers(model):
    macro_params = (
            list(model.module.net_rgb.parameters()) +
            list(model.module.net_depth.parameters()) +
            list(model.module.net_cat.parameters()) +
            list(model.module.macro_decode_heads.parameters()) +
            list(model.module.macro_norm.parameters())
    )
    cgm_params = (
            list(model.module.cgm_head.parameters()) +
            list(model.module.macro_latent_norm.parameters())
    )

    opt = torch.optim.Adam([
        {"params": macro_params, "lr": 5e-5},
        {"params": cgm_params, "lr": 5e-5},
    ], weight_decay=5e-4)

    scheduler_macro = PartialLRScheduler(
        torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99),
        group_indices=[0]
    )
    scheduler_cgm = PartialLRScheduler(
        torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=15, min_lr=1e-6),
        group_indices=[1]
    )

    return opt, scheduler_macro, scheduler_cgm


if __name__ == "__main__":
    SEED = 42
    EPOCHS = 500
    BATCH_SIZE = 16 #64
    g = set_seed(SEED)

    best_path = "CHECKPOINTS/joint_best_rgbd_load.pt"
    best_path_micro = "CHECKPOINTS/joint_best_rgbd_micro_load.pt"

    # TRAIN_TXT = "../cgmacros/lunch_by_subject_combined_kcal_train.txt"
    # TEST_TXT = "../cgmacros/lunch_by_subject_combined_kcal_test.txt"

    TRAIN_TXT = "../cgmacros/lunch_sub_train_test_val/train.txt"
    VAL_TXT = "../cgmacros/lunch_sub_train_test_val/val.txt"
    TEST_TXT = "../cgmacros/lunch_sub_train_test_val/test.txt"
    #
    # TRAIN_TXT = "../cgmacros/lunch_dinner_train_val_test/train.txt"
    # VAL_TXT = "../cgmacros/lunch_dinner_train_val_test/val.txt"
    # TEST_TXT = "../cgmacros/lunch_dinner_train_val_test/test.txt"

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
    criterion = nn.SmoothL1Loss(beta=1.0) #weighted_cgm_loss
    # opt = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=5e-4)
    # # scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)  # as RGBD
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     opt, mode="min", factor=0.5, patience=15, min_lr=1e-6
    # )

    opt, sched_macro, sched_cgm = define_opt_and_schedulers(model)


    train_model(model, EPOCHS, train_loader, val_loader, criterion, opt, sched_macro, sched_cgm, #scheduler,
                len_trainset=len(trainset), len_testset=len(valset), best_path=best_path, device=device,
                macro_weight=args.macro_loss_weight)

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
    criterion = nn.SmoothL1Loss(beta=1.0) #weighted_cgm_loss
    # opt = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=5e-4)
    # # scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)  # as RGBD
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     opt, mode="min", factor=0.5, patience=15, min_lr=1e-6
    # )

    opt, sched_macro, sched_cgm = define_opt_and_schedulers(model)


    train_model(model, EPOCHS, train_loader, val_loader, criterion, opt, sched_macro, sched_cgm, #scheduler,
                len_trainset=len(trainset), len_testset=len(valset), best_path=best_path_micro, device=device,
                macro_weight=args.macro_loss_weight)

    y_std, y_mean = trainset.stats["std"]["iAUC_log"], trainset.stats["mean"]["iAUC_log"]
    eval_model(model, best_path_micro, device, y_std, y_mean, test_loader)