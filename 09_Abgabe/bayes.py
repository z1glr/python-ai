import pandas as pd

class NaiveBayes:
    df_data: pd.DataFrame
    lst_s_results: list[str]
    
    dct_f_products: dict[float]

    f_laplace_alpha: float

    def __init__(self, df_data: pd.DataFrame, lst_s_results: list[str], f_laplace_alpha: float = 1):
        self.df_data = df_data
        self.lst_s_results = lst_s_results
        self.f_laplace_alpha = f_laplace_alpha

    def f_predict(self, str_result: str, sr_data: pd.Series, state=True) -> float:
        f_res_a = self.f_prop_P_product(sr_data, str_result, state)
        f_res_b = 0.0

        lst_results = self.lst_s_results.copy()
        lst_results.pop(self.lst_s_results.index(str_result))

        for rr in lst_results:
            f_res_b += self.f_prop_P_product(sr_data, rr, state)

        return 1 / (1 + f_res_b / f_res_a)

    def f_prop_P_product(self, sr_data: pd.Series, str_result: str, state=True) -> float:
        df_subdata = self.df_data[self.df_data[str_result] == state]
        f_result = self.df_data[str_result].mean()

        for pp in sr_data.items():
            i_count = self.f_laplace_alpha

            if pp[1] in df_subdata[pp[0]].values:
                i_count += df_subdata[pp[0]].value_counts()[pp[1]]

            zwischen = i_count / len(df_subdata)
            f_result *= zwischen

        return f_result
