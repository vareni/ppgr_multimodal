import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

#RGB-D 
class Nutrition_RGBD(Dataset):
    def __init__(self, image_path, rgb_txt_dir, rgbd_txt_dir, glucose=False, microbiome=False, transform=None,
                 norm_cols=['Baseline_Libre', 'Age', 'BMI', 'A1c', 'HOMA', 'Insulin', 'TG', 'Cholesterol',
                            'HDL', 'Non_HDL', 'CHO_HDL_ratio', 'Fasting_BG', 'LDL', 'VLDL',
                            'total_calories', 'total_carb', 'total_protein', 'total_fat', 'Gender',
                            'Fiber',
                            'Lachnospiraceae', 'Streptococcaceae', 'Bacteroidaceae', 'Enterobacteriaceae',
                            'Prevotellaceae'
                            ],
                 train=True, stats=None
                 ):  #TODO fiber is explicitly macronutrient

        file_rgb = open(rgb_txt_dir, 'r')
        file_rgbd = open(rgbd_txt_dir, 'r')
        lines_rgb = file_rgb.readlines()
        lines_rgbd = file_rgbd.readlines()
        self.images = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []
        self.images_rgbd = []
        self.glucose = glucose
        self.microbiome = microbiome

        if glucose:
            self.Fiber = []
            self.Baseline_Libre = []
            self.Age = []
            self.Gender = []
            self.BMI = []
            self.A1c = []
            self.HOMA = []
            self.Insulin = []
            self.TG = []
            self.Cholesterol =[]
            self.HDL = []
            self.Non_HDL = []
            self.LDL = []
            self.VLDL = []
            self.CHO_HDL_ratio = []
            self.Fasting_BG = []
            self.iAUC = []
            self.iAUC_log = []
            self.AUC = []
            self.AUC_log = []

            if microbiome:
                self.Lachnospiraceae = []
                self.Streptococcaceae = []
                self.Bacteroidaceae = []
                self.Enterobacteriaceae = []
                self.Prevotellaceae = []

        for line in lines_rgb:
            image_rgb = line.split()[0]
            label = line.strip().split()[1]
            calories = line.strip().split()[2]
            mass = line.strip().split()[3]
            fat = line.strip().split()[4]
            carb = line.strip().split()[5]
            protein = line.strip().split()[6]

            self.images += [os.path.join(image_path, image_rgb)]
            self.labels += [str(label)]
            self.total_calories += [np.array(float(calories))]
            self.total_mass += [np.array(float(mass))]
            self.total_fat += [np.array(float(fat))]
            self.total_carb += [np.array(float(carb))]
            self.total_protein += [np.array(float(protein))]

            if glucose:
                f = float(line.strip().split()[7])
                if not np.isfinite(f):
                    f = 0.0
                self.Fiber += [f]
                self.Baseline_Libre += [np.array(float(line.strip().split()[8]))]
                self.Age += [np.array(float(line.strip().split()[9]))]
                self.Gender += [np.array(float(line.strip().split()[10]))]
                self.BMI += [np.array(float(line.strip().split()[11]))]
                self.A1c += [np.array(float(line.strip().split()[12]))]
                self.HOMA += [np.array(float(line.strip().split()[13]))]
                self.Insulin += [np.array(float(line.strip().split()[14]))]
                self.TG += [np.array(float(line.strip().split()[15]))]
                self.Cholesterol += [np.array(float(line.strip().split()[16]))]
                self.HDL += [np.array(float(line.strip().split()[17]))]
                self.Non_HDL += [np.array(float(line.strip().split()[18]))]
                self.LDL += [np.array(float(line.strip().split()[19]))]
                self.VLDL += [np.array(float(line.strip().split()[20]))]
                self.CHO_HDL_ratio += [np.array(float(line.strip().split()[21]))]
                self.Fasting_BG += [np.array(float(line.strip().split()[22]))]

                iAUC = np.array(float(line.strip().split()[28]))
                self.iAUC += [iAUC]
                self.iAUC_log += [np.log(iAUC)]

                AUC = np.array(float(line.strip().split()[29]))
                self.AUC += [AUC]
                self.AUC_log += [np.log(AUC)]
                # self.AUC += [np.array(float(line.strip().split()[29]))]

                if microbiome:
                    self.Lachnospiraceae += [np.array(float(line.strip().split()[23]))]
                    self.Streptococcaceae += [np.array(float(line.strip().split()[24]))]
                    self.Bacteroidaceae += [np.array(float(line.strip().split()[25]))]
                    self.Enterobacteriaceae += [np.array(float(line.strip().split()[26]))]
                    self.Prevotellaceae += [np.array(float(line.strip().split()[27]))]

        for line in lines_rgbd:
            image_rgbd = line.split()[0]
            self.images_rgbd += [os.path.join(image_path, image_rgbd)]

        self.transform = transform

        self.stats = self.normalize_attrs(norm_cols + ['iAUC_log', 'AUC_log'], stats=stats, train=train)


    def normalize_attrs(self, norm_cols, stats=None, train=True, eps=1e-6):
        arrays = {}
        for col in norm_cols:
            x = getattr(self, col)
            x = np.asarray(x, dtype=np.float32)
            x = x.reshape(-1)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            arrays[col] = x

        if train:
            mean = {c: float(arrays[c].mean()) for c in norm_cols}
            std = {c: float(arrays[c].std()) for c in norm_cols}
            std = {c: (s if s > eps else 1.0) for c, s in std.items()}
            stats = {"mean": mean, "std": std}
        else:
            if stats is None:
                raise ValueError("Pass stats from train for val/test.")

        for col in norm_cols:
            x = arrays[col]
            x = (x - stats["mean"][col]) / stats["std"][col]
            setattr(self, col, x)

        return stats


    #RGB-D  20210805
    def my_loader(path, Type):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                if Type == 3:
                    img = img.convert('RGB')
                elif Type == 1:
                    img = img.convert('L')
                return img

    def __getitem__(self, index):
        img_rgb = cv2.imread(self.images[index])
        img_rgbd = cv2.imread(self.images_rgbd[index])
        try:
            img_rgb = Image.fromarray(cv2.cvtColor(img_rgb,cv2.COLOR_BGR2RGB)) # cv2转PIL
            img_rgbd = Image.fromarray(cv2.cvtColor(img_rgbd,cv2.COLOR_BGR2RGB)) # cv2转PIL
        except:
            print("corrupt img: ",self.images[index])

        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
            img_rgbd = self.transform(img_rgbd)

        if not self.glucose:
            return img_rgb, self.labels[index], self.total_calories[index], self.total_mass[index], \
                self.total_fat[index], self.total_carb[index], self.total_protein[index], img_rgbd
        elif not self.microbiome:
            return img_rgb, self.labels[index], self.total_calories[index], self.total_mass[index], \
                self.total_fat[index], self.total_carb[index], self.total_protein[index], img_rgbd, \
                self.Fiber[index], self.Baseline_Libre[index], self.Age[index], self.Gender[index], \
                self.BMI[index], self.A1c[index], self.HOMA[index], self.Insulin[index], self.TG[index], \
                self.Cholesterol[index], self.HDL[index], self.Non_HDL[index], self.LDL[index], \
                self.VLDL[index], self.CHO_HDL_ratio[index], self.Fasting_BG[index], \
                self.iAUC[index], self.iAUC_log[index], self.AUC_log[index]
                # self.AUC[index]

        return img_rgb, self.labels[index], self.total_calories[index], self.total_mass[index], \
            self.total_fat[index], self.total_carb[index], self.total_protein[index], img_rgbd, \
            self.Fiber[index], self.Baseline_Libre[index], self.Age[index], self.Gender[index], \
            self.BMI[index], self.A1c[index], self.HOMA[index], self.Insulin[index], self.TG[index], \
            self.Cholesterol[index], self.HDL[index], self.Non_HDL[index], self.LDL[index], \
            self.VLDL[index], self.CHO_HDL_ratio[index], self.Fasting_BG[index], \
            self.Lachnospiraceae[index], self.Streptococcaceae[index], self.Bacteroidaceae[index], \
            self.Enterobacteriaceae[index], self.Prevotellaceae[index], \
            self.iAUC[index], self.iAUC_log[index], self.AUC_log[index]
            # self.AUC[index]

    def __len__(self):
        return len(self.images)
    
