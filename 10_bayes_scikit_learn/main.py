from pathlib import Path
import copy
from sklearn.naive_bayes import CategoricalNB
import pandas as pd

PTH_DATA_SET = Path("diagnosis.csv")
F_TRAINING_TEST_SPLIT = 0.85
TPL_DISEASES = ("Inflammation of urinary bladder", "Nephritis of renal pelvis origin")

F_LAPLACE_ALPHA = 1
F_TRESHOLD_ERROR = 0.5

TPL_F_TEMP_BOUNDS = (33.9, 35.3, 37.8, 38.3, 39.4, 42.2)
# TPL_S_TEMP_LABELS = ("low", "normal", "mild fever", "moderate fever", "high fever")
TPL_S_TEMP_LABELS = (0, 1, 2, 3, 4)

def main():
    df_data = pd.read_csv(PTH_DATA_SET, encoding="utf_16_le")

    # map the temperature into the classes
    df_data["Temperature of patient"] = pd.cut(
        df_data["Temperature of patient"],
        bins=TPL_F_TEMP_BOUNDS,
        include_lowest=True,
        labels=TPL_S_TEMP_LABELS
    )

    df_training, df_test = df_training_test_split(df_data, F_TRAINING_TEST_SPLIT)

    cnb1 = CategoricalNB(alpha=F_LAPLACE_ALPHA)
    cnb2 = copy.deepcopy(cnb1)

    df_training_classes, df_training_categories = df_class_categorie_split(df_training, TPL_DISEASES)

    cnb1.fit(df_training_classes, df_training_categories[TPL_DISEASES[0]])
    cnb2.fit(df_training_classes, df_training_categories[TPL_DISEASES[1]])

    df_test_classes = df_test.drop(list(TPL_DISEASES), axis=1)
    res1 = cnb1.predict(df_test_classes)
    res2 = cnb2.predict(df_test_classes)

    for rr, mm in zip(res1, df_training_categories.to_numpy()):
        print (
            f"{TPL_DISEASES[0]}: {rr * 100:.1f} %"
            f"-> {rr >= F_TRESHOLD_ERROR} ({mm[0]})"
        )

    for rr, mm in zip(res2, df_training_categories.to_numpy()):
        print (
            f"{TPL_DISEASES[1]}: {rr * 100:.1f} %"
            f"-> {rr >= F_TRESHOLD_ERROR} ({mm[1]})"
        )

    # for i_count, (i_index, sr_test) in enumerate(df_test.iterrows()):
    #     sr_symptoms = sr_test.drop(list(TPL_DISEASES))

    #     res1 = cnb1.predict(sr_symptoms)
    #     print (
    #         f"{TPL_DISEASES[0]}: {res1 * 100:.1f} %"
    #         f"-> {res1 >= F_TRESHOLD_ERROR} ({sr_test[TPL_DISEASES[0]]})"
    #     )

    #     res2 = cnb2.predict(sr_symptoms)
    #     print (
    #         f"{TPL_DISEASES[1]}: {res2 * 100:.1f} %"
    #         f"-> {res2 >= F_TRESHOLD_ERROR} ({sr_test[TPL_DISEASES[1]]})"
    #     )

    # cnb.predict()

def df_training_test_split(
        df_data: pd.DataFrame,
        f_split: float
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into a training- and test-dataframe

    Args:
        df_data (pd.DataFrame): complete dataframe
        f_split (float): fraction of the training-dataframe

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: training-dataframe, test-dataframe
    """
    # randomly select the training-data
    df_training = df_data.sample(frac=f_split)
    # use the remaining data as the test-data
    df_test = df_data.drop(df_training.index)

    return df_training, df_test

def df_class_categorie_split(df_data: pd.DataFrame, lst_categories: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_classes = df_data.drop(list(lst_categories), axis=1)
    df_categories = df_data.drop(df_classes.columns, axis=1)

    return df_classes, df_categories

if __name__ == "__main__":
    main()
