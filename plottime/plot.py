import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict
import awkward as ak
import numpy as np


def plot(
    Y: ak.Array,
    X: Optional[ak.Array] = None,
    axs: Optional[List[plt.Axes]] = None,
    sharex: bool = True,
    figsize: Optional[Tuple] = None,
    dpi: Optional[int] = None,
    subplots_kwargs: Optional[Dict] = None,
    **kwargs
):
    if subplots_kwargs is None:
        subplots_kwargs = dict()
    if axs is None:
        _, axs = plt.subplots(
            nrows=len(Y[0]),  # number of signals
            ncols=1,
            sharex=sharex,
            squeeze=False,
            figsize=figsize,
            dpi=dpi,
            **subplots_kwargs
        )
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            if X is None:
                axs[j][0].plot(np.asarray(Y[i][j]), **kwargs)
            else:
                axs[j][0].plot(X[i][j], Y[i][j], **kwargs)
    return axs


