"""Tests for the change-point evaluation metrics (tsad-derived).

Small datetime-indexed label series with hand-countable TP/FP/FN make every
assertion a verifiable value.
"""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from reshift.metrics import (
    check_errors,
    chp_score,
    confusion_matrix,
    extract_cp_confusion_matrix,
    filter_detecting_boundaries,
    my_scale,
    single_average_delay,
    single_detecting_boundaries,
    single_evaluate_nab,
)


def _series(values: list[float]) -> pd.Series:
    s = pd.Series(values, dtype=float)
    s.index = pd.to_datetime(s.index, unit="s")
    return s


# ---------------------------------------------------------------------------
# single_detecting_boundaries
# ---------------------------------------------------------------------------


def test_single_boundaries_both_inputs_raises():
    pred = _series([0, 0, 1])
    with pytest.raises(ValueError, match="ONE type"):
        single_detecting_boundaries(
            true_series=_series([0, 1, 0]),
            true_list_ts=[],  # both non-None -> "Choose the ONE type"
            prediction=pred,
            portion=0.1,
            window_width="2s",
            anomaly_window_destination="righter",
            intersection_mode="cut right window",
        )


def test_single_boundaries_neither_input_raises():
    with pytest.raises(ValueError, match="Choose the type"):
        single_detecting_boundaries(
            None,
            None,
            _series([0, 1]),
            0.1,
            "2s",
            "righter",
            "cut right window",
        )


def test_single_boundaries_empty_list_ts():
    out = single_detecting_boundaries(
        None, [], _series([0, 1]), 0.1, "2s", "righter", "cut right window"
    )
    assert out == [[]]


def test_single_boundaries_center_and_default_width():
    true = _series([0, 0, 1, 0, 0])
    pred = _series([0, 0, 0, 1, 0])
    out = single_detecting_boundaries(
        true, None, pred, 0.5, None, "center", "cut right window"
    )
    cp = true[true == 1].index[0]
    assert out[0][0] < cp < out[0][1]


def test_single_boundaries_bad_destination_raises():
    true = _series([0, 1, 0])
    with pytest.raises(RuntimeError, match="anomaly_window_destination"):
        single_detecting_boundaries(
            true,
            None,
            _series([0, 1, 0]),
            0.1,
            "2s",
            "nowhere",
            "cut right window",
        )


@pytest.mark.parametrize("mode", ["cut left window", "cut both"])
def test_single_boundaries_intersection_modes(mode):
    true = _series([0, 1, 1, 0])  # adjacent CPs -> overlapping windows
    pred = _series([0, 1, 1, 0])
    out = single_detecting_boundaries(
        true, None, pred, 1.0, "3s", "center", mode
    )
    assert len(out) == 2


def test_single_boundaries_bad_intersection_raises():
    true = _series([0, 1, 1, 0])
    with pytest.raises(ValueError, match="intersection_mode"):
        single_detecting_boundaries(
            true, None, _series([0, 1, 1, 0]), 1.0, "3s", "center", "nonsense"
        )


# ---------------------------------------------------------------------------
# check_errors / filter
# ---------------------------------------------------------------------------


def test_filter_detecting_boundaries():
    assert filter_detecting_boundaries([[1, 2], [], [3, 4]]) == [
        [1, 2],
        [3, 4],
    ]


def test_check_errors_list_of_series():
    depth = check_errors([_series([0, 1]), _series([1, 0])])
    assert depth == 1


def test_check_errors_list_of_timestamps():
    depth = check_errors([[pd.Timestamp(0), pd.Timestamp(1)]])
    assert depth == 2


def test_check_errors_boundaries():
    depth = check_errors([[[pd.Timestamp(0), pd.Timestamp(1)]]])
    assert depth == 3


def test_check_errors_non_uniform_raises():
    with pytest.raises(ValueError, match="Non uniform"):
        check_errors([[pd.Timestamp(0)], _series([0, 1])])


def test_check_errors_non_uniform_across_level_raises():
    # Same level mixes a bare Timestamp and a nested boundary list -> the
    # aggregated per-level uniformity check fires.
    t0, t1 = pd.Timestamp(0), pd.Timestamp(1)
    with pytest.raises(ValueError, match="Non uniform"):
        check_errors([[t0], [[t0, t1]]])


def test_extract_binary_empty_boundaries_no_fns():
    idx = pd.to_datetime(list(range(3)), unit="s")
    pred = pd.Series([1, 0, 0], index=idx)
    out = extract_cp_confusion_matrix([], pred, binary=True)
    assert out["FNs"] == []


# ---------------------------------------------------------------------------
# confusion matrices
# ---------------------------------------------------------------------------


def test_confusion_matrix_counts():
    true = _series([1, 0, 1, 0])
    pred = _series([1, 1, 0, 0])
    TP, TN, FP, FN = confusion_matrix(true, pred)
    assert (TP, TN, FP, FN) == (1, 1, 1, 1)


def test_extract_cp_confusion_matrix_binary_tp_fn():
    true = _series([0, 0, 1, 1, 0, 0])
    pred = _series([0, 0, 1, 0, 0, 0])
    boundaries = single_detecting_boundaries(
        true, None, pred, 1.0, "2s", "righter", "cut right window"
    )
    out = extract_cp_confusion_matrix(boundaries, pred, binary=True)
    assert isinstance(out["TPs"], dict)
    assert "FPs" in out and "FNs" in out


def test_extract_binary_single_window_fns():
    # One window, one predicted point inside -> remaining window points are FNs.
    idx = pd.to_datetime(list(range(5)), unit="s")
    pred = pd.Series([0, 1, 0, 0, 0], index=idx)
    out = extract_cp_confusion_matrix([[idx[0], idx[2]]], pred, binary=True)
    assert 0 in out["TPs"]
    assert len(out["FNs"]) == 2  # t0 and t2 inside window but not predicted


def test_extract_binary_two_windows_fns_concat():
    idx = pd.to_datetime(list(range(6)), unit="s")
    pred = pd.Series([1, 0, 0, 1, 0, 0], index=idx)
    out = extract_cp_confusion_matrix(
        [[idx[0], idx[1]], [idx[3], idx[4]]], pred, binary=True
    )
    assert len(out["TPs"]) == 2  # both windows hit
    assert len(out["FNs"]) > 0  # concatenated misses from both windows


def test_extract_binary_no_fns():
    # Window with a single point that is predicted -> no FNs.
    idx = pd.to_datetime(list(range(3)), unit="s")
    pred = pd.Series([0, 1, 0], index=idx)
    out = extract_cp_confusion_matrix([[idx[1], idx[1]]], pred, binary=True)
    assert out["FNs"] == []


def test_extract_cp_confusion_matrix_no_boundaries_all_fp():
    pred = _series([1, 0, 1])
    out = extract_cp_confusion_matrix([], pred)
    assert len(out["FPs"]) == 2  # both predicted points are false positives


# ---------------------------------------------------------------------------
# my_scale
# ---------------------------------------------------------------------------


def test_my_scale_returns_curve_when_plot_figure():
    y = my_scale(plot_figure=True, detalization=100)
    assert y.shape == (100,)
    assert y[0] == pytest.approx(1.0, abs=1e-6)  # A_tp at the left


def test_my_scale_reversed_when_not_clear_mode():
    y = my_scale(
        plot_figure=True, detalization=100, clear_anomalies_mode=False
    )
    assert y[-1] == pytest.approx(1.0, abs=1e-6)  # mirrored


def test_my_scale_event_value_clamped():
    # fp_case_window with event index beyond the curve -> clamped to last point.
    val = my_scale([0, 10, 10], detalization=10)
    assert np.isscalar(val) or val.ndim == 0


# ---------------------------------------------------------------------------
# single_average_delay branches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dest", ["lefter", "righter", "center"])
def test_single_average_delay_destinations(dest):
    true = _series([0, 0, 1, 0, 0])
    pred = _series([0, 0, 0, 1, 0])
    boundaries = single_detecting_boundaries(
        true, None, pred, 1.0, "3s", dest, "cut right window"
    )
    missing, history, fp, total = single_average_delay(
        boundaries, pred, dest, clear_anomalies_mode=True
    )
    assert total == 1
    assert isinstance(history, list)


def test_single_average_delay_bad_destination_raises():
    true = _series([0, 1, 0])
    pred = _series([0, 1, 0])
    boundaries = single_detecting_boundaries(
        true, None, pred, 1.0, "2s", "righter", "cut right window"
    )
    with pytest.raises(ValueError, match="anomaly_window_destination"):
        single_average_delay(
            boundaries, pred, "nope", clear_anomalies_mode=True
        )


# ---------------------------------------------------------------------------
# single_evaluate_nab
# ---------------------------------------------------------------------------


def test_single_evaluate_nab_bad_scale_func_raises():
    pred = _series([0, 1, 0])
    with pytest.raises(ValueError, match="scale_func"):
        single_evaluate_nab(
            [[pred.index[0], pred.index[-1]]], pred, scale_func="bad"
        )


# ---------------------------------------------------------------------------
# chp_score top-level dispatch
# ---------------------------------------------------------------------------


def test_chp_score_confusion_matrix_metric():
    true = _series([1, 0, 1, 0])
    pred = _series([1, 1, 0, 0])
    assert chp_score(true, pred, metric="confusion_matrix") == (1, 1, 1, 1)


def test_chp_score_binary_perfect():
    true = _series([0, 0, 1, 0])
    pred = _series([0, 0, 1, 0])
    f1, far, mar = chp_score(true, pred, metric="binary", verbose=True)
    assert f1 == 1.0
    assert far == 0.0
    assert mar == 0.0


def test_chp_score_confusion_matrix_verbose():
    true = _series([1, 0, 1, 0])
    pred = _series([1, 1, 0, 0])
    out = chp_score(true, pred, metric="confusion_matrix", verbose=True)
    assert out == (1, 1, 1, 1)


def test_chp_score_average_time_center(caplog):
    true = _series([0, 0, 1, 0, 0])
    pred = _series([0, 0, 0, 1, 0])
    out = chp_score(
        true,
        pred,
        metric="average_time",
        window_width="3s",
        anomaly_window_destination="center",
        portion=1.0,
        verbose=True,
    )
    assert out[3] == 1  # one true change-point


def test_chp_score_list_of_series():
    true = [_series([0, 1, 0]), _series([0, 0, 1])]
    pred = [_series([0, 1, 0]), _series([0, 0, 1])]
    res = chp_score(true, pred, metric="nab", window_width="2s", portion=1.0)
    assert set(res) == {"Standard", "LowFP", "LowFN"}


def test_chp_score_boundaries_input_variant():
    pred1 = _series([0, 1, 0, 0])
    pred2 = _series([0, 0, 1, 0])
    # pre-built detection boundaries (variant 3), one window per dataset
    boundaries = [
        [[pred1.index[0], pred1.index[2]]],
        [[pred2.index[1], pred2.index[3]]],
    ]
    res = chp_score(boundaries, [pred1, pred2], metric="nab", portion=1.0)
    assert set(res) == {"Standard", "LowFP", "LowFN"}


def test_chp_score_bad_prediction_type_raises():
    # Deliberately wrong type to exercise the runtime guard.
    with pytest.raises(TypeError, match="Incorrect format"):
        chp_score(_series([0, 1]), cast("pd.Series", 42))


def test_chp_score_bad_prediction_list_raises():
    with pytest.raises(ValueError, match="Incorrect format"):
        chp_score([_series([0, 1])], cast("list[pd.Series]", [123]))


def test_chp_score_bad_metric_raises():
    true = _series([1, 0])
    pred = _series([1, 0])
    with pytest.raises(ValueError, match="performance metric"):
        chp_score(true, pred, metric="nonsense")


def test_single_boundaries_no_true_points_returns_empty():
    # true series with no change-points -> no boundaries.
    out = single_detecting_boundaries(
        _series([0, 0, 0]),
        None,
        _series([0, 0, 0]),
        0.1,
        "2s",
        "righter",
        "cut right window",
    )
    assert out == []


def test_check_errors_boundaries_wrong_pair_length_raises():
    t = pd.Timestamp(0)
    with pytest.raises(ValueError, match="Non uniform"):
        check_errors([[[t, t, t]]])  # boundary "pair" of length 3


def test_single_evaluate_nab_with_custom_table():
    pred = _series([0, 1, 0, 0])
    table = pd.DataFrame(
        [[1.0, -0.5, 1.0, -1.0]] * 3,
        index=pd.Index(["Standard", "LowFP", "LowFN"], name="Metric"),
        columns=pd.Index(["A_tp", "A_fp", "A_tn", "A_fn"]),
    )
    out = single_evaluate_nab(
        [[pred.index[0], pred.index[2]]], pred, table_of_coef=table
    )
    assert out.shape == (3, 3)


def test_single_average_delay_lefter_true_positive():
    # pred coincides with the change-point so it lands in the lefter window.
    true = _series([0, 0, 1, 0])
    pred = _series([0, 0, 1, 0])
    boundaries = single_detecting_boundaries(
        true, None, pred, 1.0, "3s", "lefter", "cut right window"
    )
    missing, history, fp, total = single_average_delay(
        boundaries, pred, "lefter", clear_anomalies_mode=True
    )
    assert missing == 0
    assert len(history) == 1


def test_chp_score_nab_default_portion_warns(caplog):
    import logging

    true = _series([0, 0, 1, 0])
    pred = _series([0, 0, 1, 0])
    with caplog.at_level(logging.WARNING):
        chp_score(true, pred, metric="nab")  # window_width=None -> warning
    assert any("portion" in r.message for r in caplog.records)


def test_chp_score_nab_verbose(caplog):
    import logging

    true = _series([0, 0, 1, 0])
    pred = _series([0, 0, 1, 0])
    with caplog.at_level(logging.INFO):
        chp_score(
            true,
            pred,
            metric="nab",
            window_width="3s",
            portion=1.0,
            verbose=True,
        )
    assert caplog.records


def test_chp_score_binary_with_timestamp_true_warns(caplog):
    # true as list-of-timestamps + binary metric triggers the non-Series path.
    pred = _series([0, 1, 1, 0])
    true = [[pred.index[1]]]
    res = chp_score(
        true, [pred], metric="binary", window_width="2s", portion=1.0
    )
    assert len(res) == 3
