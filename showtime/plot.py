import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy.typing as npt
import numpy as np
from typing import Optional
from saxex.utils import (
    NORM_BINS,
)
from saxex.utils import zscore_transform, zscore_inverse_transform, sax_to_norm_mapper
import awkward as ak


def plot_mts(
    X: ak.Array,
    Y: Optional[ak.Array] = None,
    axs: Optional[list[plt.Axes]] = None,
    sharex: bool = True,
    figsize: Optional[tuple] = None,
    dpi: Optional[int] = None,
    subplots_kwargs: Optional[dict] = dict(),
    **kwargs
):
    if axs is None:
        _, axs = plt.subplots(
            nrows=len(X[0]),
            ncols=1,
            sharex=sharex,
            squeeze=False,
            figsize=figsize,
            dpi=dpi,
            **subplots_kwargs
        )
    for i in range(len(X)):
        for j in range(len(X[i])):
            if Y is None:
                axs[j][0].plot(X[i][j], **kwargs)
            else:
                axs[j][0].plot(Y[i][j], X[i][j], **kwargs)
    return axs


def plot_subsequence(
    s: np.array,
    y: np.array,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple] = None,
    dpi: Optional[int] = None,
    subplots_kwargs: Optional[dict] = dict(),
    **kwargs
):
    if ax is None:
        plt.figure(figsize=figsize, dpi=dpi, **subplots_kwargs)
        ax = plt.gca()
    ax.plot(y, s, **kwargs)
    return ax


def plot_sax_vlines(
    start: np.array,
    end: np.array,
    indices: np.array,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple] = None,
    dpi: Optional[int] = None,
    subplots_kwargs: Optional[dict] = dict(),
    color: str = "gray",
    linestyle: str = "--",
    alpha: float = 1,
    **kwargs
):
    if ax is None:
        plt.figure(figsize=figsize, dpi=dpi, **subplots_kwargs)
        ax = plt.gca()
    for idx in start:
        ax.axvline(
            indices[idx], alpha=alpha, linestyle=linestyle, color=color, **kwargs
        )
    ax.axvline(
        indices[end[-1] - 1], alpha=alpha, linestyle=linestyle, color=color, **kwargs
    )
    return ax


def plot_sax_hlines(
    bins: np.array,
    indices: np.array,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple] = None,
    dpi: Optional[int] = None,
    subplots_kwargs: Optional[dict] = dict(),
    color: str = "gray",
    linestyle: str = "--",
    alpha: float = 1,
    **kwargs
):
    if ax is None:
        plt.figure(figsize=figsize, dpi=dpi, **subplots_kwargs)
        ax = plt.gca()
    for bin_edge in bins:
        ax.hlines(
            bin_edge,
            xmin=indices[0],
            xmax=indices[-1],
            alpha=alpha,
            color=color,
            linestyle=linestyle,
            **kwargs
        )
    return ax


def plot_sax_symbols(
    s: np.array,
    sax_vector: np.array,
    start: np.array,
    end: np.array,
    indices: np.array,
    bins_centroids: np.array,
    nan_symbol: str = "*",
    text_y_relative_offset: float = 0.01,
    color: str = "black",
    fontweight: str = "bold",
    verticalalignment: str = "baseline",
    horizontalalignment: str = "center",
    fontsize: str = "large",
    path_effect_linewidth: int = 4,
    path_effect_foreground: str = "white",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple] = None,
    dpi: Optional[int] = None,
    subplots_kwargs: Optional[dict] = dict(),
    **kwargs
):
    if ax is None:
        plt.figure(figsize=figsize, dpi=dpi, **subplots_kwargs)
        ax = plt.gca()
    for i, (idx_start, idx_end) in enumerate(zip(start, end)):
        ax.text(
            x=np.mean([indices[idx_end - 1], indices[idx_start]]),
            y=bins_centroids[i]
            + (text_y_relative_offset * (np.nanmax(s) - np.nanmin(s)))
            if not np.isnan(bins_centroids[i])
            else np.nanmean(s),
            s=nan_symbol if np.isnan(sax_vector[i]) else str(int(sax_vector[i])),
            color=color,
            fontweight=fontweight,
            verticalalignment=verticalalignment,
            horizontalalignment=horizontalalignment,
            fontsize=fontsize,
            path_effects=[
                pe.withStroke(
                    linewidth=path_effect_linewidth, foreground=path_effect_foreground
                )
            ],
            **kwargs
        )
    return ax


def plot_sax(
    s: np.array,
    sax_vector: np.array,
    start: np.array,
    end: np.array,
    indices: np.array,
    bins: np.array,
    bins_centroids: np.array,
    plot_vlines: bool = True,
    plot_hlines: bool = True,
    plot_symbols: bool = True,
    nan_symbol: str = "*",
    text_y_relative_offset: float = 0.01,
    axvline_color: str = "gray",
    axvline_style: str = "--",
    axvline_alpha: float = 1,
    axvline_kwargs: dict = dict(),
    axhline_color: str = "gray",
    axhline_style: str = "--",
    axhline_alpha: float = 1,
    axhline_kwargs: dict = dict(),
    color: str = "black",
    fontweight: str = "bold",
    verticalalignment: str = "baseline",
    horizontalalignment: str = "center",
    fontsize: str = "large",
    path_effect_linewidth: int = 4,
    path_effect_foreground: str = "white",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple] = None,
    dpi: Optional[int] = None,
    subplots_kwargs: Optional[dict] = dict(),
    **kwargs
):
    if ax is None:
        plt.figure(figsize=figsize, dpi=dpi, **subplots_kwargs)
        ax = plt.gca()
    if plot_vlines:
        ax = plot_sax_vlines(
            start=start,
            end=end,
            indices=indices,
            alpha=axvline_alpha,
            linestyle=axvline_style,
            color=axvline_color,
            ax=ax,
            **axvline_kwargs
        )
    if plot_hlines:
        ax = plot_sax_hlines(
            bins=bins,
            indices=indices,
            ax=ax,
            alpha=axhline_alpha,
            color=axhline_color,
            linestyle=axhline_style,
            **axhline_kwargs
        )
    if plot_symbols:
        ax = plot_sax_symbols(
            s=s,
            sax_vector=sax_vector,
            start=start,
            end=end,
            indices=indices,
            bins_centroids=bins_centroids,
            nan_symbol=nan_symbol,
            text_y_relative_offset=text_y_relative_offset,
            color=color,
            fontweight=fontweight,
            verticalalignment=verticalalignment,
            horizontalalignment=horizontalalignment,
            fontsize=fontsize,
            path_effect_linewidth=path_effect_linewidth,
            path_effect_foreground=path_effect_foreground,
            ax=ax,
            figsize=figsize,
            dpi=dpi,
            subplots_kwargs=subplots_kwargs,
            **kwargs
        )
    return ax


# def plot_paa(x: npt.ArrayLike, paa_array: npt.ArrayLike, start: npt.ArrayLike, end: npt.ArrayLike,
#              ax: Optional[plt.Axes] = None, color="C1", **kwargs) -> plt.Axes:
#     if ax is None:
#         ax = plt.gca()
#     ax.plot(x, **kwargs)
#     for i, (idx_start, idx_end) in enumerate(zip(start, end)):
#         ax.plot(range(idx_start, idx_end), np.repeat(paa_array[i], idx_end - idx_start), color=color)
#     return ax


# def plot_sax(
#         x: np.array, sax_array: npt.ArrayLike,
#         bins: npt.ArrayLike,
#         bins_centroids: npt.ArrayLike,
#         start: npt.ArrayLike,
#         end: npt.ArrayLike,
#         ax: Optional[plt.Axes] = None,
#         color="C1",
#         **kwargs) -> plt.Axes:
#     alpha = 0.2
#     style = "--"
#     color_lines = "gray"
#     if ax is None:
#         ax = plt.gca()
#     for idx in start:
#         ax.axvline(idx, alpha=alpha, linestyle=style, color=color_lines)
#     ax.axvline(end[-1], alpha=alpha, linestyle=style, color=color_lines)
#     ax.plot(x, **kwargs)
#     for bin_edge in bins:
#         ax.axhline(bin_edge, color=color_lines, linestyle=style)
#     for i, (idx_start, idx_end) in enumerate(zip(start, end)):
#         ax.plot(range(idx_start, idx_end), np.repeat(bins_centroids[i], idx_end - idx_start), color=color)
#         ax.text(
#             x=np.mean([idx_end, idx_start]),
#             y=bins_centroids[i] + (0.01 * (x.max() - x.min())) if not np.isnan(bins_centroids[i]) else np.nanmean(x),
#             s="*" if np.isnan(sax_array[i]) else str(int(sax_array[i])),
#             color=color,
#             fontweight="bold",
#             verticalalignment="baseline",
#             horizontalalignment="center",
#             fontsize="large",
#             path_effects=[pe.withStroke(linewidth=4, foreground="white")]
#         )
#     return ax


def plot_sax_normalized(
    x: np.array,
    sax_array: npt.ArrayLike,
    alphabet_size: int,
    start: npt.ArrayLike,
    end: npt.ArrayLike,
    ax: Optional[plt.Axes] = None,
    color="C1",
    **kwargs
) -> plt.Axes:
    ax = plot_sax(
        x=zscore_transform(x),
        sax_array=sax_array,
        bins=NORM_BINS[str(alphabet_size)],
        bins_centroids=sax_to_norm_mapper(
            sax_sequence=sax_array, alphabet_size=alphabet_size
        ),
        start=start,
        end=end,
        ax=ax,
        color=color,
        **kwargs
    )
    return ax


def plot_sax_denormalized(
    x: np.array,
    sax_array: npt.ArrayLike,
    alphabet_size: int,
    start: npt.ArrayLike,
    end: npt.ArrayLike,
    ax: Optional[plt.Axes] = None,
    color="C1",
    **kwargs
) -> plt.Axes:
    ax = plot_sax(
        x=x,
        sax_array=sax_array,
        bins=zscore_inverse_transform(np.array(NORM_BINS[str(alphabet_size)]), x),
        bins_centroids=zscore_inverse_transform(
            sax_to_norm_mapper(sax_sequence=sax_array, alphabet_size=alphabet_size), x
        ),
        start=start,
        end=end,
        ax=ax,
        color=color,
        **kwargs
    )
    return ax


# def plot_sax(x: np.array, sax_array: npt.ArrayLike, alphabet_size: int, start: npt.ArrayLike, end: npt.ArrayLike,
#              ax: Optional[plt.Axes] = None, color="C1", **kwargs) -> plt.Axes:
#     alpha = 0.2
#     style = "--"
#     color_lines = "gray"
#     if ax is None:
#         ax = plt.gca()
#     for idx in start:
#         ax.axvline(idx, alpha=alpha, linestyle=style, color=color_lines)
#     ax.axvline(end[-1], alpha=alpha, linestyle=style, color=color_lines)
#     ax.plot(zscore_transform(x), **kwargs)
#     bins = NORM_BINS[str(alphabet_size)]
#     for bin_edge in bins:
#         ax.axhline(bin_edge, color=color_lines, linestyle=style)
#     bins_approx_values = sax_to_norm_mapper(sax_sequence=sax_array, alphabet_size=alphabet_size)
#     for i, (idx_start, idx_end) in enumerate(zip(start, end)):
#         ax.plot(range(idx_start, idx_end), np.repeat(bins_approx_values[i], idx_end - idx_start), color=color)
#         ax.text(
#             x=np.mean([idx_end, idx_start]),
#             y=bins_approx_values[i] + (0.05 * (x.max() - x.min())) if not np.isnan(bins_approx_values[i]) else 0,
#             s="*" if np.isnan(sax_array[i]) else str(int(sax_array[i])),
#             color=color,
#             fontweight="bold",
#             verticalalignment="baseline",
#             horizontalalignment="center",
#             fontsize="large",
#             path_effects=[pe.withStroke(linewidth=4, foreground="white")]
#         )
#     return ax


if __name__ == "__main__":
    X = ak.Array([[np.random.random(size=100)] * 10])
    plot_mts(X, figsize=(20, 5))
    plt.show()
    # from saxex.utils.utils import segmentation_indexes, paa, digitize
    #
    # alphabet_size = 5
    # sequence = np.random.random(100)
    # start, end = segmentation_indexes(sequence.size, 16)
    # paa_vector = paa(zscore_transform(sequence), start, end)
    # bins = NORM_BINS[str(alphabet_size)]
    # sax_vector = digitize(paa_vector, bins=bins)
    # sax_vector[2:5] = np.nan
    # fig = plt.figure()
    # plot_sax_normalized(sequence, sax_vector, alphabet_size, start, end)
    # plt.show()
    # plot_sax_denormalized(sequence, sax_vector, alphabet_size, start, end)
    # plt.show()
