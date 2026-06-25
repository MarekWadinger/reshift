"""Tests for Hankel embedding, normalization, and polynomial features."""

import numpy as np
import pandas as pd

from reshift.preprocessing import hankel, normalize, polynomial_extension


def test_normalize_minmax():
    out = normalize(np.array([0.0, 5.0, 10.0]))
    assert np.allclose(out, [0.0, 0.5, 1.0])


def test_normalize_ignores_nan():
    out = normalize(np.array([np.nan, 2.0, 4.0]))
    assert np.isnan(out[0])
    assert np.allclose(out[1:], [0.0, 1.0])


def test_hankel_noop_when_order_le_one():
    X = np.array([1.0, 2.0, 3.0])
    # hn <= 1 returns the input unchanged.
    assert hankel(X, 1) is X


def test_hankel_drop_partial_rows():
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = hankel(X, 3, return_partial=False)
    expected = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=float)
    assert np.array_equal(out, expected)


def test_hankel_nan_partial():
    X = np.array([1.0, 2.0, 3.0])
    out = hankel(X, 2, return_partial=True)
    # First row's earliest delay is unknown -> NaN.
    assert np.isnan(out[0, 0])
    assert np.array_equal(out[1:], [[1, 2], [2, 3]])


def test_hankel_dataframe_keeps_columns_and_index():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=pd.Index([10, 11, 12]))
    out = hankel(X, 2, return_partial="copy")
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["a_0", "a_1"]
    assert list(out.index) == [10, 11, 12]


def test_polynomial_extension_degree_two():
    df = pd.DataFrame({"a": [2.0], "b": [3.0]})
    out = polynomial_extension(df, 2)
    # degree 1: a, b ; degree 2: a*a, a*b, b*b
    assert list(out.columns) == ["a", "b", "a*a", "a*b", "b*b"]
    assert out.iloc[0].tolist() == [2.0, 3.0, 4.0, 6.0, 9.0]
