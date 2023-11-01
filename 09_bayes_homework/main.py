from pathlib import Path
import pandas as pd

from bayes import NaiveBayes

PTH_DATA_FILE = Path("diagnosis.data")
S_CSV_ENCODING = "utf_16_le"
CHR_CSV_DELIMITER = '\t'
CHR_CSV_DECIMAL_POINT = ','
F_TRAINING_TEST_SPLIT = 0.8
F_LAPLACE_ALPHA = 1

F_TRESHOLD_ERROR = 0.5

TPL_COLUMN_NAMES = ("Temperature of patient", "Occurrence of nausea", "Lumbar  ain", "Urine pushing", "Micturition pains", "Burning of urethra, itch, swelling of urethra outlet", "Inflammation of urinary bladder", "Nephritis of renal pelvis origin")
TPL_DISEASES = ("Inflammation of urinary bladder", "Nephritis of renal pelvis origin")

TPL_F_TEMP_BOUNDS = (35.0, 36.5, 37.4, 42.0)
TPL_S_TEMP_LABELS = ("cold", "normal", "hot")

def main():
    # load the dataset into a pandas dataframe
    df_data = pd.read_csv(PTH_DATA_FILE, delimiter=CHR_CSV_DELIMITER, names=TPL_COLUMN_NAMES, decimal=CHR_CSV_DECIMAL_POINT, encoding=S_CSV_ENCODING)
    df_data: pd.DataFrame = df_data.map(yes_no_map)

    df_data["Temperature of patient"] = pd.cut(df_data["Temperature of patient"], bins=TPL_F_TEMP_BOUNDS, include_lowest=True, labels=TPL_S_TEMP_LABELS)

    # df_data.drop("Temperature of patient", axis=1)

    # split the data into the training- and test-data
    df_training, df_test = df_training_test_split(df_data, F_TRAINING_TEST_SPLIT)

    # initialize the naive bayes
    nb_acute_inflammation = NaiveBayes(df_training, list(TPL_DISEASES), F_LAPLACE_ALPHA)

    for ii, sr_test in df_test.iterrows():
        sr_symptoms = sr_test.drop(list(TPL_DISEASES))

        for str_disease in TPL_DISEASES:
            f_res = nb_acute_inflammation.f_predict(str_disease, sr_symptoms)

            if (f_res >= F_TRESHOLD_ERROR) ^ sr_test[str_disease]:
                print(f"{sr_test.name} {str_disease} {f_res:.4f} {sr_test[str_disease]}")

def yes_no_map(datapoint):
    match datapoint:
        case "yes":
            return True
        case "no":
            return False
        case _:
            return datapoint

def df_training_test_split(df_data: pd.DataFrame, f_split: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into a training- and test-dataframe

    Args:
        df_data (pd.DataFrame): complete dataframe
        f_split (float): fraction of the training-dataframe

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: training-dataframe, test-dataframe
    """
    df_training = df_data.sample(frac=f_split)
    df_test = df_data.drop(df_training.index)

    return df_training, df_test

if __name__ == "__main__":
    main()
