import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict
import awkward as ak
import numpy as np
from matplotlib.collections import LineCollection


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
        axs = build_axs(
            n_signals=len(Y[0]),
            dpi=dpi,
            figsize=figsize,
            sharex=sharex,
            **subplots_kwargs
        )
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            ax = axs[j][0]
            y = np.asarray(Y[i][j])
            x = X[i][j] if X is not None else None
            plot_signal(x=x, y=y, ax=ax, **kwargs)
    return axs


def plot_saliency(
    Y: ak.Array,
    S: ak.Array,
    X: Optional[ak.Array] = None,
    cmap="viridis",
    norm=None,
    axs: Optional[List[plt.Axes]] = None,
    sharex: bool = True,
    figsize: Optional[Tuple] = None,
    dpi: Optional[int] = None,
    subplots_kwargs: Optional[Dict] = None,
    linewidth: int = 1,
    alpha: float = 1,
    **kwargs
):
    if subplots_kwargs is None:
        subplots_kwargs = dict()
    if axs is None:
        axs = build_axs(len(Y[0]), dpi, figsize, sharex, **subplots_kwargs)
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            ax = axs[j][0]
            y = np.asarray(Y[i][j])
            x = np.asarray(X[i][j]) if X is not None else None
            s = np.asarray(S[i][j])
            plot_saliency_signal(
                y=y,
                s=s,
                x=x,
                alpha=alpha,
                ax=ax,
                cmap=cmap,
                linewidth=linewidth,
                norm=norm,
                **kwargs
            )
    return axs


def plot_saliency_signal(
    y, s, ax, x=None, cmap="viridis", linewidth=1, norm=None, alpha: float = 1, **kwargs
):
    plot_signal(x=x, ax=ax, y=y, alpha=0)
    if x is None:
        points = np.array([range(len(y)), y]).T.reshape(-1, 1, 2)
    else:
        points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha, **kwargs
    )
    lc.set_array(s)
    ax.add_collection(lc)
    return ax


def plot_signal(y, ax, x=None, **kwargs):
    if x is None:
        ax.plot(y, **kwargs)
    else:
        ax.plot(x, y, **kwargs)
    return ax


def build_axs(n_signals, dpi, figsize, sharex, **kwargs):
    _, axs = plt.subplots(
        nrows=n_signals,
        ncols=1,
        sharex=sharex,
        squeeze=False,
        figsize=figsize,
        dpi=dpi,
        **kwargs
    )
    return axs
