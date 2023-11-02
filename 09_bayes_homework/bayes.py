"""Implementation of the bayes classificator"""
import pandas as pd

class NaiveBayes:
    """Implementation of a naive bayes classificator"""
    # the dataset used to predict results
    df_data: pd.DataFrame
    # column-names for the result-columns
    lst_s_results: list[str]

    # alpha-factor for the laplace-correction
    f_laplace_alpha: float

    def __init__(self, df_data: pd.DataFrame, lst_s_results: list[str], f_laplace_alpha: float = 1):
        """initializing function

        Args:
            df_data (pd.DataFrame): dataframe for predicting the results
            lst_s_results (list[str]): column-names for the result-columns
            f_laplace_alpha (float, optional): factor for the laplace-correction. Defaults to 1.
        """
        self.df_data = df_data
        self.lst_s_results = lst_s_results
        self.f_laplace_alpha = f_laplace_alpha

    def f_predict(self, str_result: str, sr_data: pd.Series, state=True) -> float:
        """predicts the propability for a result for the given input states

        Args:
            str_result (str): name of the result column
            sr_data (pd.Series): input-data
            state (bool, optional): state of the result to predict for. Defaults to True.

        Returns:
            float: propability for the result given the input states
        """
        # calculate the prior-propabilty
        f_prior = self.df_data[str_result].mean()
        # calculate the prior * likelihood
        f_res_a = f_prior * self.f_likelihood(str_result, sr_data, state)
        # calculate the part of the evidence
        f_res_b = (1 - f_prior) * self.f_likelihood(str_result, sr_data, not state)

        # calculate the naive-bayes
        return 1 / (1 + f_res_b / f_res_a)

    def f_likelihood(self, str_result: str, sr_data: pd.Series, state=True) -> float:
        """calculate the likelihood of a result given the input states

        Args:
            str_result (str): name of the result column
            sr_data (pd.Series): input-data
            state (any, optional): state of the result to predict for. Defaults to True.

        Returns:
            float: likelihood of the result given the input states
        """
        # select only the data, where the result is of the state we are looking for
        df_subdata = self.df_data[self.df_data[str_result] == state]
        f_result = 1

        # go through the individual input-states
        for pp in sr_data.items():
            # initialize with the laplace-alpha
            i_count = self.f_laplace_alpha

            # check, wether the value even exists in the data
            if pp[1] in df_subdata[pp[0]].values:
                # add the number of occurences to the counter
                i_count += df_subdata[pp[0]].value_counts()[pp[1]]

            # multiply the individual propability to the total propability
            f_result *= i_count / len(df_subdata)

        return f_result
