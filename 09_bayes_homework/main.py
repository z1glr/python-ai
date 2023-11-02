"""
using a naive bayes classificator to predict diseases
in the acute inflammation dataset
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from bayes import NaiveBayes

# path to the dataset
PTH_DATA_FILE = Path("diagnosis.data")
# encoding of the dataset
S_CSV_ENCODING = "utf_16_le"
# value-delimiter of the csv-dataset
CHR_CSV_DELIMITER = '\t'
# decimal-point of the dataset
CHR_CSV_DECIMAL_POINT = ','
# split between training and test data
F_TRAINING_TEST_SPLIT = 0.8
# value for the alpha value of the laplace factor
F_LAPLACE_ALPHA = 1

# treshold, at which a result should be true
F_TRESHOLD_ERROR = 0.5

# names of the csv-columns
TPL_COLUMN_NAMES = (
	"Temperature of patient",
	"Occurrence of nausea",
	"Lumbar  ain",
	"Urine pushing",
	"Micturition pains",
	"Burning of urethra, itch, swelling of urethra outlet",
	"Inflammation of urinary bladder",
	"Nephritis of renal pelvis origin"
)
# names of the disease-columns
TPL_DISEASES = ("Inflammation of urinary bladder", "Nephritis of renal pelvis origin")

# bounds for the temperature-classes
TPL_F_TEMP_BOUNDS = (35.0, 36.5, 38, 42.0)
# names of the temperature classes
TPL_S_TEMP_LABELS = ("cold", "normal", "hot")

def main():
    """main-function
    """
    # load the dataset into a pandas dataframe
    df_data = pd.read_csv(
        filepath_or_buffer=PTH_DATA_FILE,
        delimiter=CHR_CSV_DELIMITER,
        names=TPL_COLUMN_NAMES,
        decimal=CHR_CSV_DECIMAL_POINT,
        encoding=S_CSV_ENCODING,
        true_values=["yes"],
        false_values=["no"]
    )

    # map the temperature into the classes
    df_data["Temperature of patient"] = pd.cut(
        df_data["Temperature of patient"],
        bins=TPL_F_TEMP_BOUNDS,
        include_lowest=True,
        labels=TPL_S_TEMP_LABELS
    )

    # split the data into the training- and test-data
    df_training, df_test = df_training_test_split(df_data, F_TRAINING_TEST_SPLIT)

    # initialize the naive bayes
    nb_acute_inflammation = NaiveBayes(df_training, list(TPL_DISEASES), F_LAPLACE_ALPHA)

    # create an empty numpy array to store the results
    np_results = np.empty((len(df_test), 5))

    # go throught the test-data and test every single dataset
    for i_count, (i_index, sr_test) in enumerate(df_test.iterrows()):
        # take only the symptoms
        sr_symptoms = sr_test.drop(list(TPL_DISEASES))

        # print the index (= line-number) of the data in the csv
        print(f"\nline {i_index}")

        # dictionary to store the results of the current data
        dct_res = {}

        # go through all the diseases
        for str_disease in TPL_DISEASES:
            # predigt the propability of the current disease given the symptoms
            f_res = nb_acute_inflammation.f_predict(str_disease, sr_symptoms)

            print (
                f"{str_disease}: {f_res * 100:.1f} %"
                f"-> {f_res >= F_TRESHOLD_ERROR} ({sr_test[str_disease]})"
            )

            # store the results in the dictionary of the current disease
            dct_res[str_disease] = (sr_test[str_disease], f_res)

        # store the current diseases results in the results-array
        np_results[i_count] = (i_index, *dct_res[TPL_DISEASES[0]], *dct_res[TPL_DISEASES[1]])

    # construct the header of the results-dataframe
    tpl_header_values = ("reference", "propability")
    tpl_header_diseases = (x for x in TPL_DISEASES for _ in range(len(tpl_header_values)))
    tpl_header = tuple(x for x in zip(tpl_header_diseases, tpl_header_values * len(TPL_DISEASES)))

    # store the results in a dataframe
    df_results = pd.DataFrame(
        np_results[:, 1:],
        index=np_results[:, 0],
        columns=pd.MultiIndex.from_tuples(tpl_header, names=("disease", "values"))
    )

    # change the result-columns into booleans
    for dd in TPL_DISEASES:
        df_results[dd, "reference"] = df_results[dd, "reference"].astype(bool)

    # change the index into an integer
    df_results.index = df_results.index.astype(int)

    # create a subplot to show all the plots at once
    axs = plt.subplots(len(TPL_DISEASES), 2)[1]

    # create plots for every disease
    for xx, dd in enumerate(TPL_DISEASES):
        # create a plot for the two possible results
        for yy, rr in enumerate((False, True)):
            # select only the data of the current disease and result
            df_sub = df_results[df_results[dd, "reference"] == rr]

            # ticks for the x-axis
            x_ticks = np.arange(len(df_sub))
            # create the plot
            axs[xx, yy].bar(x_ticks, height=df_sub[dd, "propability"])

            # format the plot
            axs[xx, yy].set_xticks(x_ticks)
            # set the labels of x-axis to the column numbers
            axs[xx, yy].set_xticklabels(df_sub.index)
            axs[xx, yy].set_ylim((0, 1))
            axs[xx, yy].title.set_text(f"{dd} = {rr}")
            axs[xx, yy].set_xlabel("index")
            axs[xx, yy].set_ylabel("propability")

    plt.show()

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

if __name__ == "__main__":
    main()
