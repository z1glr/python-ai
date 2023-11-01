import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing

from matplotlib import pyplot as plt

from kNearestNeighbor import KNearestNeighbor

F_FRAC = 0.8

I_SPLIT_TRAIN = int(150 * F_FRAC)
I_SPLIT_VERIFY = 150 - I_SPLIT_TRAIN

def main():
    DF_DATA = pd.read_csv(Path("iris_set.csv"), quotechar='"', delimiter=',')

    DF_DATA = DF_DATA.sample(frac=1)

    lst_i_k = range(1, I_SPLIT_TRAIN)

    lst_args = zip([DF_DATA] * len(lst_i_k), lst_i_k)

    with multiprocessing.Pool(32) as p:
        res = p.map(test_with_k, lst_args)

    for rr in res:
        print (f"{rr[0]}-nearest-neighbors: errors = {rr[1]}")

    np_res = 1 - np.array(res).T / len(DF_DATA) * 2

    plt.plot(*np_res)
    plt.show()

def test_with_k(args) -> int:
    kn = KNearestNeighbor(args[0][:I_SPLIT_TRAIN], ["sepal.length","sepal.width","petal.length","petal.width"], args[1], "variety")

    i_wrong_count = 0

    for _, ss in args[0][I_SPLIT_VERIFY:].iterrows():
        res_predigt = kn.predict(ss)
        
        if res_predigt != ss["variety"]:
            i_wrong_count += 1

    return (args[1], i_wrong_count)

if __name__ == "__main__":
    main()
