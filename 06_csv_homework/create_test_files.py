"""path handling"""
from pathlib import Path
import random
import tempfile
import shutil
import pandas as pd
import numpy as np

STUDENTS_TOTAL = 100
FILE_COUNT = 10
RANGE_TASKS = (1, 10)
RANGE_POINTS = (0, 5)
RANGE_TASK_ATTENDENCE = (0.4, 0.9)
OUT_DIR = Path("csv")
DELIMITER = ';'

DF_NAMES = pd.read_csv(Path("names.csv"), delimiter=';')

def main():
    # setup the output directory
    create_empty_dir(OUT_DIR)

    lst_students: list[tuple[str, str, int]] = []

    set_unique_numbers: set[int] = set()

    # create unique numbers for every student
    while len(set_unique_numbers) < STUDENTS_TOTAL:
        set_unique_numbers.add(random.randint(10000, 99999))

    for ii in set_unique_numbers:
        names = s_get_random_name().values.tolist()[0]
        names.append(ii)

        lst_students.append(names)

    df_students = pd.DataFrame(lst_students, columns=["name", "vorname", "Matr-Nr"])

    for _ in range(FILE_COUNT):
        attendance = random.random()
        attendance *= (RANGE_TASK_ATTENDENCE[1] - RANGE_TASK_ATTENDENCE[0])
        attendance += RANGE_TASK_ATTENDENCE[0]

        df_scores = create_scores(df_students.sample(frac=attendance), random.randint(*RANGE_TASKS), RANGE_POINTS)

        store_df_to_file(df_scores)


def create_empty_dir(directory: Path) -> None:
    """_summary_
    ensures, that a specified path is an empty dir, deletes files if necessary
    
    Args:
        directory (Path): _description_ path of the directory
    """
    # if the path exists, delete the directory
    if directory.exists():
        shutil.rmtree(directory)

    # create the dir
    OUT_DIR.mkdir()

def get_score(max: int) -> int:
    return random.randint(0, max)

def s_get_random_name() -> pd.DataFrame:
    name = DF_NAMES.sample(n = 1)

    return name

def create_scores(df_students: pd.DataFrame, i_count_tasks: int, tpl_i_point_range: tuple[int, int]):
    df_points = np.random.default_rng().integers(*tpl_i_point_range, (len(df_students), i_count_tasks))

    columns = [f"A{ii + 1}" for ii in range(i_count_tasks)]

    df_points = pd.DataFrame(df_points, columns=columns, index=df_students.index)

    return pd.concat([df_students, df_points], axis=1)

def store_df_to_file(data: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", encoding="utf-8", dir=OUT_DIR, delete=False) as out_file:
        data.to_csv(out_file, sep=DELIMITER, lineterminator='\n', index=False)

if __name__ == "__main__":
    main()