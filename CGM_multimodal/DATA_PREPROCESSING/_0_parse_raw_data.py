import os
import pandas as pd
import numpy as np
import itertools


def areaUnderCurve(a, b):
    total = 0
    temp = 0
    for i in range(len(a)-1):
        if (b[i+1]-b[0]>=0) and (b[i]-b[0]>=0):
            temp = ((b[i]-b[0]+b[i+1]-b[0])/2)*(a[i+1]-a[i])
        elif (b[i+1]-b[0] < 0) and (b[i]-b[0] >= 0):
            temp = (b[i]-b[0])*((b[i]-b[0])/(b[i]-b[i+1])*(a[i+1]-a[i])/2)
        elif (b[i+1]-b[0] >= 0) and (b[i]-b[0] < 0):
            temp = (b[i+1]-b[0])*((b[i+1]-b[0])/(b[i+1]-b[i])*(a[i+1]-a[i])/2)
        elif (b[i]-b[0] < 0) and (b[i+1]-b[0] < 0):
            temp = 0
        total = total + temp
    return total

def calc_iauc(cgm, sampling_interval):
    a = []
    for i in range(len(cgm)):
        a.append(i * sampling_interval[i])
    return areaUnderCurve(a, cgm)

def calc_auc(cgm, sampling_interval):
    return np.trapz(cgm, dx=sampling_interval)

def gather_data(meal_types=["lunch", "dinner"], get_all_meals=False):
    data_all_sub = pd.DataFrame(columns = ["sub", "Libre GL", "Carb", "Protein", "Fat", "Fiber", "Image path", "Meal Type", "Amount Consumed"])

    hours = 2
    libre_samples = hours * 4 + 1
    
    for sub in sorted(os.listdir("../../CGMacros")):
        if sub[:8] != "CGMacros":
            continue
        data = pd.read_csv(os.path.join("../../CGMacros", sub, sub+'.csv'))
        data.columns = data.columns.str.strip()            
        data_sub = pd.DataFrame(columns = ["sub", "Timestamp", "Libre GL", "Carb", "Protein", "Fat", "Fiber", "Image path", "Meal Type", "Amount Consumed"])
        potential_data = data[~data["Meal Type"].isna()]
        if not get_all_meals:
            meal_types = [m.strip().lower() for m in meal_types]
            potential_data = data[
                data["Meal Type"].fillna("").str.strip().str.lower().isin(meal_types)
            ]
        for index in potential_data.index:
        # for index in data[(data["Meal Type"] == "Lunch") | (data["Meal Type"] == "lunch") | (data["Meal Type"] == "dinner") | (data["Meal Type"] == "Dinner")].index:
        # for index in data[~data["Meal Type"].isna()].index:
            data_meal = {}
            data_meal["sub"] = sub[-3:]
            data_meal["Libre GL"] = data["Libre GL"][index:index+135:15].to_list()
            data_meal["Libre GL before meal"] = data["Libre GL"].iloc[index - 29:index+1].to_list()
            if len(data_meal["Libre GL"]) < 9:
                continue
            data_meal["iAUC"] = calc_iauc(data_meal["Libre GL"], [15 for i in range(libre_samples)])
            data_meal["AUC"] = calc_auc(data_meal["Libre GL"], 15)
            data_meal["Carb"] = data["Carbs"][index] * 4
            data_meal["Protein"] = data["Protein"][index] * 4
            data_meal["Fat"] = data["Fat"][index] * 9
            data_meal["Fiber"] = data["Fiber"][index] * 2
            data_meal["Calories"] = data["Calories"][index]
            data_meal["Image path"] = data["Image path"][index]
            # if (data["Meal Type"][index] == "Lunch") or (data["Meal Type"][index] == "lunch"):
            #     data_meal["Meal Type"] = "Lunch"
            # else:
            #     data_meal["Meal Type"] = "Dinner"
            data_meal["Meal Type"] = (data["Meal Type"][index]).strip().lower()
            if "Amount Consumed" in data.columns:
                data_meal["Amount Consumed"] = data["Amount Consumed"][index]
            else:
                data_meal["Amount Consumed"] = 1.0
            data_meal["Timestamp"] = data["Timestamp"][index]
            data_sub = data_sub.append(data_meal, ignore_index=True)
        if data_sub["Carb"].iloc[0] == 24 and data_sub["Protein"].iloc[0] == 22 and data_sub["Fat"].iloc[0] == 10.5 and data_sub["Fiber"].iloc[0] == 0.0:
            data_sub = data_sub.iloc[1:]
        data_all_sub = data_all_sub.append(data_sub, ignore_index=True)    

    print(f"Gathered data rows: {data_all_sub.shape[0]}")
    return data_all_sub

def add_bio_info(data_all_sub):
    df = pd.read_csv("../../CGMacros/bio.csv")

    a1c = df["A1c PDL (Lab)"].dropna().to_numpy()
    fasting_glucose = df["Fasting GLU - PDL (Lab)"].dropna().to_numpy()
    fasting_insulin = df["Insulin "].dropna().to_numpy() # in uIU/mL (ideal range: 2.6 - 24.9)
    fasting_insulin = [float(str(x).strip(' (low)')) for x in fasting_insulin]
    
    HOMA = (fasting_insulin * fasting_glucose)/405
    
    tg = df["Triglycerides"].dropna().to_numpy()
    cholesterol = df["Cholesterol"].dropna().to_numpy()
    HDL = df["HDL"].dropna().to_numpy()
    non_HDL = df["Non HDL "].dropna().to_numpy()
    ldl = df["LDL (Cal)"].dropna().to_numpy()
    vldl = df["VLDL (Cal)"].dropna().to_numpy()
    cho_hdl_ratio = df["Cho/HDL Ratio"].dropna().to_numpy()
    
    # patients = []
    # for i in range(len(a1c)):
    #     if a1c[i] < 5.7:
    #         patients.append("H")
    #     if a1c[i] >= 5.7 and a1c[i] <=6.4:
    #         patients.append("P")
    #     if a1c[i] > 6.4:
    #         patients.append("T2D")
    # patients = np.array(patients)
    
    # h_index = np.where(patients == "H")[0]
    # p_index = np.where(patients == "P")[0]
    # t_index = np.where(patients == "T2D")[0]

    weights = df["Body weight "].dropna().to_numpy()
    heights = df["Height "].dropna().to_numpy()
    
    weight_kg = weights * 0.453592
    
    total_heights = []
        
    for i in range(len(heights)):
        inches = heights[i]
        h = float(inches)
        total_heights.append(h * 0.0254)
    
    BMI = []
    for height, weight in zip(total_heights, weight_kg):
        bmi = weight/(height**2)
        BMI.append(bmi)
    
    BMI = np.array(BMI)
    
    age = df["Age"].to_numpy()
    gender = df["Gender"].to_list()
    gender = [1 if x == 'M'  else -1 for x in gender]

    # libre_data = data_all_sub["Libre GL"]

    interp_gl = []
    for i in range(len(data_all_sub["Libre GL"])):
        interp_gl.append(data_all_sub["Libre GL"][i][0])
    
    data_all_sub["Baseline_Libre"] = interp_gl
    data_all_sub = data_all_sub.drop(columns=["Libre GL"])
    
    subjects = data_all_sub["sub"].unique()
    
    new_age = []
    new_gender = []
    new_BMI = []
    new_a1c = []
    new_HOMA = []
    new_fasting_insulin = []
    new_tg = []
    new_cholestrol = []
    new_HDL = []
    new_non_HDL = []
    new_ldl = []
    new_vldl = []
    new_cho_hdl_ratio = []
    new_fasting_glucose = []
    
    for i in range(len(subjects)):
        match_length = len(data_all_sub[data_all_sub["sub"] == subjects[i]])
        new_age.extend([age[i]] * match_length)
        new_gender.extend([gender[i]] * match_length)
        new_BMI.extend([BMI[i]] * match_length)
        new_a1c.extend([a1c[i]] * match_length)
        new_HOMA.extend([HOMA[i]] * match_length)
        new_fasting_insulin.extend([fasting_insulin[i]] * match_length)
        new_tg.extend([tg[i]] * match_length)
        new_cholestrol.extend([cholesterol[i]] * match_length)
        new_HDL.extend([HDL[i]] * match_length)
        new_non_HDL.extend([non_HDL[i]] * match_length)
        new_ldl.extend([ldl[i]] * match_length)
        new_vldl.extend([vldl[i]] * match_length)
        new_cho_hdl_ratio.extend([cho_hdl_ratio[i]] * match_length)
        new_fasting_glucose.extend([fasting_glucose[i]] * match_length)
    
    
    data_all_sub["Age"] = new_age
    data_all_sub["Gender"] = new_gender
    data_all_sub["BMI"] = new_BMI
    data_all_sub["A1c"] = new_a1c
    data_all_sub["HOMA"] = new_HOMA
    data_all_sub["Insulin"] = new_fasting_insulin
    data_all_sub["TG"] = new_tg
    data_all_sub["Cholesterol"] = new_cholestrol
    data_all_sub["HDL"] = new_HDL
    data_all_sub["Non HDL"] = new_non_HDL
    data_all_sub["LDL"] = new_ldl
    data_all_sub["VLDL"] = new_vldl
    data_all_sub["CHO/HDL ratio"] = new_cho_hdl_ratio
    data_all_sub["Fasting BG"] = new_fasting_glucose

    data_all_sub = data_all_sub.iloc[:, 1:]
    return data_all_sub


def premeal_features(glucose_list):
    y = np.array(glucose_list, dtype=float)

    if len(y) < 30 or np.isnan(y).any():
        return pd.Series({
            "CGM_mean_30min_before": np.nan,
            "CGM_std_30min_before": np.nan,
            "CGM_delta_30min_before": np.nan,
            "CGM_slope_30min_before": np.nan,
            "CGM_mean_15min_before": np.nan,
            "CGM_std_15min_before": np.nan,
            "CGM_delta_15min_before": np.nan,
            "CGM_slope_15min_before": np.nan,
        })

    def calc_window_features(vals):
        vals = np.array(vals, dtype=float)
        x = np.arange(len(vals))  # 0, 1, 2, ...

        slope = np.polyfit(x, vals, 1)[0]

        return vals.mean(), vals.std(), vals[-1] - vals[0], slope

    y30 = y[-30:]
    mean_30, std_30, delta_30, slope_30 = calc_window_features(y30)

    y15 = y[-15:]
    mean_15, std_15, delta_15, slope_15 = calc_window_features(y15)

    return pd.Series({
        "CGM_mean_30min_before": mean_30,
        "CGM_std_30min_before": std_30,
        "CGM_delta_30min_before": delta_30,
        "CGM_slope_30min_before": slope_30,
        "CGM_mean_15min_before": mean_15,
        "CGM_std_15min_before": std_15,
        "CGM_delta_15min_before": delta_15,
        "CGM_slope_15min_before": slope_15,
    })


def parse_raw_data(save_all_path=None, save_correct_path=None, save_incorrect_path=None, get_all_meals=False,
                   meal_types=["lunch", "dinner"]):
    data_all_sub = gather_data(get_all_meals=get_all_meals, meal_types=meal_types)

    data_all_sub = data_all_sub[data_all_sub["iAUC"] > 0]
    data_all_sub = data_all_sub.dropna(subset=['Image path', 'Fiber'])
    data_all_sub.reset_index(inplace=True)

    print(f"After filters iAUC>0, dropna: {data_all_sub.shape[0]}")

    features = data_all_sub["Libre GL before meal"].apply(premeal_features)
    data_all_sub = pd.concat([data_all_sub, features], axis=1)

    data_all_sub = add_bio_info(data_all_sub)

    cols = ["Calories", "Carb", "Protein", "Fat"]
    z = data_all_sub[cols].eq(0)
    
    counts = {} 
    counts[tuple(cols)] = int(z.all(axis=1).sum())
    for k in (3, 2, 1):
        for comb in itertools.combinations(cols, k):
            other = [c for c in cols if c not in comb]
            m = z[list(comb)].all(axis=1) & (~z[other].any(axis=1))
            counts[comb] = int(m.sum())
    
    print(f"Columns with zeros:\n{counts}")

    data_all_sub = data_all_sub.loc[~(data_all_sub[cols] == 0).any(axis=1)]

    print(f"After removing zero columns: {data_all_sub.shape[0]}")

    data_all_sub["Correct Nutrition"] = True
    for i, row in data_all_sub.iterrows():
        tmp = row['Carb'] + row['Protein'] + row['Fat']
        if (abs(tmp - row['Calories']) / row['Calories']) > 0.2:
            # print(f"{row['sub']} calculations don't match: {row['Calories']} != {tmp}, img: {row['Image path']}")
            data_all_sub.loc[i, "Correct Nutrition"] = False

    print(f"Incorrect nutrition rows: {data_all_sub[data_all_sub['Correct Nutrition'] == False].shape[0]}")
    print(f"Correct nutrition rows: {data_all_sub[data_all_sub['Correct Nutrition'] == True].shape[0]}")

    if save_all_path is not None:
        data_all_sub.to_csv(save_all_path, index=False)
    if save_correct_path is not None:
        data_all_sub[data_all_sub["Correct Nutrition"] == True].to_csv(save_correct_path, index=False)
        print(f"Saved: {save_correct_path}")
    if save_incorrect_path is not None:
        data_all_sub[data_all_sub["Correct Nutrition"] == False].to_csv(save_incorrect_path, index=False)
        print(f"Saved: {save_incorrect_path}")

    return data_all_sub[data_all_sub["Correct Nutrition"] == True]


if __name__ == "__main__":

    save_all_path = '../splits/lunch_dinner/data_all_lunch_dinner_gl_stats.csv'
    save_correct_path = '../splits/lunch_dinner/data_correct_lunch_dinner_gl_stats.csv'
    save_incorrect_path = '../splits/lunch_dinner/data_incorrect_lunch_dinner_gl_stats.csv'

    data_gathered = parse_raw_data(save_all_path, save_correct_path, save_incorrect_path, get_all_meals=False,
                                  meal_types=["lunch", "dinner"])
