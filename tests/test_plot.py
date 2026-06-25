"""Tests for plotting helpers (headless Agg backend, set via conftest)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from pandas import DataFrame

from reshift.plot import is_tex_available, plot_chd, set_size

# Matches the plot_chd `datas` parameter type.
_Datas = (
    dict[str, np.ndarray | DataFrame | None]
    | list[np.ndarray | DataFrame | None]
)


def test_is_tex_available_returns_bool():
    assert isinstance(is_tex_available(), bool)


@pytest.mark.parametrize(
    ("width", "expected_pt"),
    [("article", 390.0), ("thesis", 426.79135), ("beamer", 307.28987)],
)
def test_set_size_named_widths(width, expected_pt):
    w_in, h_in = set_size(width)
    assert w_in == pytest.approx(expected_pt / 72.27 * 1.2)
    assert h_in < w_in  # golden-ratio height is shorter than width


def test_set_size_numeric_width_and_subplots():
    w_in, h_in = set_size(200.0, fraction=0.5, subplots=(2, 1))
    assert w_in > 0
    assert h_in > 0


def test_plot_chd_dict_input_with_truth():
    rng = np.random.default_rng(0)
    datas: _Datas = {
        "signal": rng.normal(size=200),
        "score": rng.normal(size=200),
    }
    fig, axs = plot_chd(
        datas, y_true=[50, 120], labels=["a", "b"], grace_period=10
    )
    assert len(axs) == 2
    plt.close(fig)


def test_plot_chd_list_with_dataframe_and_normalize():
    df = pd.DataFrame(
        {"x": np.linspace(0, 1, 100), "y": np.linspace(1, 0, 100)}
    )
    arr = np.sin(np.linspace(0, 6, 100))
    datas: _Datas = [df, arr]
    fig, axs = plot_chd(
        datas,
        labels=["multi", "single"],
        idx_start=10,
        idx_end=90,
        normalize=True,
    )
    assert axs.shape[0] == 2
    plt.close(fig)


def test_plot_chd_with_inlays():
    sig = np.sin(np.linspace(0, 10, 300))
    datas: _Datas = {"s": sig, "s2": sig * 2}
    fig, axs = plot_chd(
        datas,
        y_true=[120, 150, 250],  # two inside the inlay, one outside
        ids_in_start=[100],
        ids_in_end=[200],
        grace_period=5,
        labels=["s", "s2"],
    )
    assert len(axs) == 2
    plt.close(fig)


def test_plot_chd_inlays_without_y_true():
    sig = np.sin(np.linspace(0, 10, 300))
    datas: _Datas = {"s": sig, "s2": sig * 2}
    fig, axs = plot_chd(
        datas, ids_in_start=[100], ids_in_end=[200], labels=["s", "s2"]
    )
    assert len(axs) == 2
    plt.close(fig)


def test_plot_chd_inlays_cp_inside_without_grace():
    sig = np.sin(np.linspace(0, 10, 300))
    datas: _Datas = {"s": sig, "s2": sig * 2}
    # cp inside the inlay but grace_period falsy -> skip the dashed grace line.
    fig, axs = plot_chd(
        datas,
        y_true=[150],
        ids_in_start=[100],
        ids_in_end=[200],
        grace_period=None,
        labels=["s", "s2"],
    )
    assert len(axs) == 2
    plt.close(fig)


def test_plot_chd_y_true_without_grace():
    sig = np.sin(np.linspace(0, 10, 100))
    datas: _Datas = [sig, sig]
    # y_true drawn but grace_period falsy -> skip the dashed grace line.
    fig, axs = plot_chd(datas, y_true=[40], grace_period=None)
    assert len(axs) == 2
    plt.close(fig)


def test_plot_chd_non_axes_entry_returns_early():
    fig, ax = plt.subplots(1, 1)
    # Second "axis" is not an Axes -> the isinstance guard returns early.
    axs = np.array([ax, "not-an-axes"], dtype=object)
    datas: _Datas = [np.arange(10.0), np.arange(10.0)]
    out_fig, _ = plot_chd(datas, axs=axs)
    assert out_fig is fig
    plt.close(fig)


def test_plot_chd_into_existing_axes():
    fig, axs = plt.subplots(2, 1)
    sig = np.arange(50.0)
    datas: _Datas = [sig, sig]
    fig2, _ = plot_chd(datas, axs=axs)
    assert fig2 is fig
    plt.close(fig)


def test_plot_chd_handles_none_series():
    datas: _Datas = [None, np.arange(20.0)]
    fig, axs = plot_chd(datas)
    assert len(axs) == 2
    plt.close(fig)


def test_plot_chd_non_axes_short_circuit():
    # A single subplot returns a bare Axes (not an array); plot_chd guards on
    # the isinstance(ax, Axes) check and returns early for non-Axes entries.
    fig, ax = plt.subplots(1, 1)
    datas: _Datas = [np.arange(10.0)]
    _, out_axs = plot_chd(datas, axs=np.array([ax]))
    assert isinstance(out_axs[0], Axes)
    plt.close(fig)
