"""easy path handling"""
from pathlib import Path
import shutil
import re
import pandas as pd

CSV_DIR = Path("csv") # path of the directory with the input files from ILIAS
OUT_DIR = Path("output") # master-output-directory
# output-directory for the results of the individual students
OUT_DIR_STUDENTS = OUT_DIR / "students"
OUT_FILE_RESULTS = OUT_DIR / "results.csv" # output-file for the percentage output
DELIMITER = ';' # delimiter of the csv-files
QUOTE_CHAR = '"'
OUTPUT_STUDENT_NAMES = False # wether to output the student names or only their matricle number

I_MAT_NUM = "Matr-Nr" # column-name of the matricel number
S_FORE_NAME = "vorname" # column-name of the forename
S_SIRE_NAME = "name" # column-name of the sire-name

TRES_PASSED = 0.5 # average treshold with which the student passed the test

# regex to check for column-names of the tasks
RE_NUMBERS = re.compile(r"^(?P<identifier>A)(?P<number>\d+)$")

class DuplicateMatrNrException(Exception):
    """_summary_ Exception where the same matricle-number appears multiple times
    in the same dataframe
    """

def main():
    """_summary_ main function
    """
    # delete the output directory and its contents
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    dct_dataframes_from_csv = {}

    # create the output directory
    OUT_DIR_STUDENTS.mkdir(exist_ok=True, parents=True)
    OUT_FILE_RESULTS.parent.mkdir(exist_ok=True, parents=True)

    # read the individual csv files into dataframes and store them into a list
    for ff in CSV_DIR.rglob("*.csv"):
        dct_dataframes_from_csv[ff] = df_load_csv(ff)

    # combine the individual dataframes into a single big one
    df_master = df_merge_dataframes_from_list(list(dct_dataframes_from_csv.values()))

    # find the maximum points for every task
    df_max_points = df_find_max_points_per_task(df_master)

    # output the points of the individual studens
    for ii, dd in df_master.iterrows():
        export_student_row(dd, ii)

    # output the percentage-results of all the students
    export_students_percent(df_master, df_max_points)

def df_load_csv(csv_path: Path) -> pd.DataFrame:
    """_summary_ load a csv file into a dictionary

    Args:
        csv_path (Path): _description_ path to the csv file

    Returns:
        pd.DataFrame: _description_ dataframe with the contents of the csv file
    """

    # index_col: column-names to be used to identify the individual rows
    return pd.read_csv(
        filepath_or_buffer=csv_path,
        delimiter=DELIMITER,
        quotechar=QUOTE_CHAR,
        index_col=[I_MAT_NUM, S_SIRE_NAME, S_FORE_NAME]
    )

def df_merge_dataframes_from_list(lst_datafarmes: list[pd.DataFrame]) -> pd.DataFrame:
    """_summary_ combine a list full of dataframes into a single big one

    Args:
        lst_datafarmes (list[pd.DataFrame]): _description_ list of the individual dataframes

    Returns:
        pd.DataFrame: _description_ single master dataframe
    """
    # counter for the homework
    i_counter = 1

    # list with the individual dataframes but with new column names
    lst_renamed_dataframes = []

    # go through the individual dataframes and rename them
    for dd in lst_datafarmes:
        # create a rename map for the column names
        dct_rename_map = dct_create_rename_map(dd.columns, i_counter)

        # create a copy of the dataframe with renamed columns
        df_copy = dd.rename(columns=dct_rename_map)

        lst_renamed_dataframes.append(df_copy)

        # IMPORTANT: increase homework-number-counter
        i_counter += 1

    # concat the individual renamed dataframes into a single one along the columns
    try:
        df_master = pd.concat(objs=lst_renamed_dataframes, axis="columns")
    except pd.errors.InvalidIndexError as e:
        print (
            "invalid index, possibleindex duplicates in a single homework, please check manually"
        )

        raise e

    # check wether a matrical-number appears multiple times
    sr_index_matr_nr = df_master.index.droplevel([1, 2])

    # create a boolean map, wether the index has a duplicate
    sr_index_matr_nr_duplicates = sr_index_matr_nr.duplicated(keep=False)

    # extract the indexes from all the duplicates from the master-dataframe
    lst_duplicates = list(df_master[sr_index_matr_nr_duplicates].index)

    # if there are any entries in the duplicate list, print them and then raise an exception
    if len(lst_duplicates) > 0:
        # format the matricle-number and student-names into a string each
        lst_str_duplicats = list(map(lambda tt: f"{tt[0]}: {tt[1]}, {tt[2]}", lst_duplicates))

        # raise an exception
        raise DuplicateMatrNrException(
            "The same matricle number appears mutiple times with a different name"
            f"({lst_str_duplicats})"
        )

    # return the master dataframe
    return df_master

def dct_create_rename_map(lst_column_names: list[str], i_identifier) -> dict[str, str]:
    """_summary_ create a rename map for the column names so we can distinguish the tasks from 
    the individual homeworks

    Args:
        lst_column_names (list[str]): _description_ list of the individual tasks of the homework
        i_identifier (_type_): _description_ number of the homework

    Returns:
        dict[str, str]: _description_ rename map with the original task and the new task names
    """
    dct_map = {}

    for nn in lst_column_names:
        # check wether the current column is a task
        re_res = RE_NUMBERS.match(nn)

        if re_res:
            dct_map[nn] = f"{re_res.group('identifier')}{i_identifier}.{re_res.group('number')}"

    return dct_map

def df_find_max_points_per_task(df_points: pd.DataFrame) -> pd.DataFrame:
    """_summary_ returns the maximum points obtained in every single task

    Args:
        df_points (pd.DataFrame): _description_ dataframe with all the points from all the students

    Returns:
        pd.DataFrame: _description_ dataframe with all the maximum points
    """
    df_max_points = df_points.max()

    return df_max_points

def export_student_row(sr_points: pd.Series, t_ident: tuple[int, str, str]):
    """_summary_ write a give student with its points into a csv file

    Args:
        df_points (pd.DataFrame): _description_ points of the student
        t_ident (tuple[int, str, str]): _description_ identifier of the student
    """
    # construct the path of the output file
    out_file = (OUT_DIR_STUDENTS / str(t_ident[0])).with_suffix(".csv")

    # drop all the NaN values from the timeseries
    # (from the homeworks the student didn't participate in)
    sr_points.dropna(inplace=True)

    # check wether the student names should be outputted or only the matricel number
    if OUTPUT_STUDENT_NAMES:
        header = True # set the default value of True which outputs the complete header
    else:
        header = [t_ident[0]] # specify to only output the matricel number

    # write the timeseries into a file
    sr_points.to_csv(path_or_buf=out_file, sep=DELIMITER, header=header)

def export_students_percent(df_points: pd.DataFrame, sr_max_points: pd.Series):
    """_summary_ export the points-fraction for every task into a csv-file

    Args:
        df_points (pd.DataFrame): _description_ dataframe with all the points
        from every student at every task
        sr_max_points (pd.Series): _description_ series with the maximum
        achieved points at every task
    """
    # calculate the percentage of every task of every student
    fd_percentage = df_points / sr_max_points

    # calculate the mean of every task
    sr_percentage_mean = fd_percentage.mean(axis="columns")

    # check wether the student names should be written or not
    if not OUTPUT_STUDENT_NAMES:
        # if no student names are requested, drop them form the index
        sr_percentage_mean.index = sr_percentage_mean.index.droplevel([1, 2])

    # check for every student, wether they passed the treshold
    sr_passed = pd.Series(index=sr_percentage_mean.index, dtype=bool)
    sr_passed.values[:] = True
    sr_passed.where(sr_percentage_mean >= TRES_PASSED, False, inplace=True)

    # write the points into a file, set the header accordingly
    df_out = pd.concat([sr_percentage_mean, sr_passed], axis=1)

    df_out.to_csv(path_or_buf=OUT_FILE_RESULTS, sep=DELIMITER, header=["fraction", "passed"])

if __name__ == "__main__":
    main()
