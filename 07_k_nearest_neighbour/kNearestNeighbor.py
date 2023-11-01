import math
import pandas as pd

class KNearestNeighbor:
    """simple implementation of k-nearest-neighbor

    Returns:
        _type_: _description_
    """
    df_data: pd.DataFrame
    lst_s_distance_columns: list[str]
    i_k: int
    s_class_column: str

    def __init__(self, df_train_data, lst_s_distance_columns: list[str], i_k: int, s_class_column: str):
        """_summary_

        Args:
            train_data (_type_): _description_
            distance_columns (list[str]): Name of columns for distance calculation
            k (int): the k-nearest neighbors
        """
        self.df_data = df_train_data
        self.lst_s_distance_columns = lst_s_distance_columns
        self.n = i_k
        self.s_class_column = s_class_column

    def dist_func(self, row, value) -> float:
        """a simple euclidean distance function

        Args:
            row (_type_): _description_
            value (_type_): _description_

        Returns:
            float: distance between the two points
        """
        inner_value = 0

        for k in self.lst_s_distance_columns:
            inner_value += (row[k] - value[k]) ** 2

        return math.sqrt(inner_value)

    def predict(self, value):
        """_summary_ prediction of the class of a single row

        Args:
            value (_type_): _description_

        Returns:
            _type_: _description_
        """
        dist = self.df_data.apply(lambda x: self.dist_func(x, value), axis=1)
        nn = self.df_data.loc[dist.sort_values().iloc[0:self.n].index]

        return nn[self.s_class_column].value_counts().index[0]
