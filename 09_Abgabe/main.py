from pathlib import Path
import numpy as np
import pandas as pd

PTH_DATA_FILE = Path("diagnosis.csv")
F_TRAINING_DATA = 0.8

DF_DATA: pd.DataFrame 

LST_DISEASES = ["bladder_inflammation", "nephritis"]

TPL_F_EDGES = (35.0, 36.5, 37.5, 38.1, 38.6, 39.1, 40.0, 42.0)
TPL_S_LABELS = ("cold", "normal temperature","subfebrile temperature","slight fever","moderate fever","high fever","very high fever")

def main():
    df_data = pd.read_csv(PTH_DATA_FILE, delimiter=",")

    df_data["temperature"] = pd.cut(df_data["temperature"], bins=TPL_F_EDGES, include_lowest=True, labels=TPL_S_LABELS)

    df_data.drop("temperature", axis=1, inplace=True)

    df_data_training = df_data.sample(frac=F_TRAINING_DATA)
    # df_data_training = df_data[int(len(df_data) * F_TRAINING_DATA):]
    df_data_test = df_data.drop(df_data_training.index)

    df_data_training.to_csv(Path("training.csv"))

    global DF_DATA
    DF_DATA = df_data_training

    for ii, pp in df_data_test.iterrows():
        sr_symptoms = pp.drop(LST_DISEASES)

        res = []

        for dd in LST_DISEASES:
            res.append((dd, bayes(dd, sr_symptoms), pp[dd]))
            print(f"{dd} {bayes(dd, sr_symptoms):.4f} {pp[dd]}")

        if res[0][1] > res[1][1] and res[0][2] > res[1][2] or res[0][1] < res[1][1] and res[0][2] < res[1][2]:
            # print ("good")
            ...
        else:
            # print(pp)
            print("problemo")

        print()

def bayes(i, lst_x: pd.Series):
    df_subdata = DF_DATA[DF_DATA[i] == True]

    if np.array_equal(lst_x.values, [False, True, False, False, False]):
        print ("foobar")

    res_oben =  DF_DATA[i].mean()
    res_unten = 0.0

    for xx in lst_x.items():
        res_oben *= prob_P (xx, i, df_subdata)

    for jj in LST_DISEASES:
        df_subdata = DF_DATA[DF_DATA[jj] == True]

        f_zwischen = DF_DATA[jj].mean()

        for xx in lst_x.items():
            f_zwischen *= prob_P(xx, jj, df_subdata)

        res_unten += f_zwischen

    # print(res_oben, res_unten)
    
    return res_oben / res_unten

def prob_P(x, i, data: pd.Series):
    df_subdata = data[data[i] == True]

    if x[1] in df_subdata[x[0]].values:
        # print(f"{df_subdata[x[0]].value_counts()[x[1]]}/{len(data)}")
        return df_subdata[x[0]].value_counts()[x[1]] / len(data)
    else:
        return 0.0000001

if __name__ == "__main__":
    main()
