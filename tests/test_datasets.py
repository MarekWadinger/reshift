"""Tests for dataset loaders with mocked network and a temp working dir.

No real downloads: ``requests`` is monkeypatched and every loader runs against
either a pre-seeded local cache or a fake HTTP response.
"""

import io

import numpy as np
import pandas as pd
import pytest

from reshift import datasets


@pytest.fixture(autouse=True)
def _chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


class _Resp:
    def __init__(
        self, *, status=200, text="", content=b"", json_data=None, headers=None
    ):
        self.status_code = status
        self.text = text
        self.content = content
        self._json = json_data
        self.headers = headers or {}

    def json(self):
        return self._json

    def iter_content(self, chunk_size=1):
        yield self.content


# ---------------------------------------------------------------------------
# load_dateset
# ---------------------------------------------------------------------------


def test_load_dateset_reads_local_file(tmp_path):
    f = tmp_path / "local.txt"
    np.savetxt(f, np.array([1.0, 2.0, 3.0]))
    out = datasets.load_dateset(str(f), "http://unused", save=False)
    assert np.allclose(out, [1.0, 2.0, 3.0])


def test_load_dateset_downloads_and_saves(monkeypatch, tmp_path):
    monkeypatch.setattr(
        datasets.requests,
        "get",
        lambda *a, **k: _Resp(text="1.0\n2.0\n\n3.0\n"),
    )
    target = tmp_path / "sub" / "remote.txt"
    out = datasets.load_dateset(str(target), "http://x", save=True)
    assert np.allclose(out, [1.0, 2.0, 3.0])
    assert target.exists()  # persisted to disk


def test_load_dateset_download_without_save(monkeypatch, tmp_path):
    monkeypatch.setattr(
        datasets.requests, "get", lambda *a, **k: _Resp(text="7.0\n8.0\n")
    )
    out = datasets.load_dateset(
        str(tmp_path / "nosave.txt"), "http://x", save=False
    )
    assert np.allclose(out, [7.0, 8.0])
    assert not (tmp_path / "nosave.txt").exists()  # not persisted


def test_load_dateset_http_error_raises(monkeypatch):
    monkeypatch.setattr(
        datasets.requests, "get", lambda *a, **k: _Resp(status=404)
    )
    with pytest.raises(ValueError, match="404"):
        datasets.load_dateset("missing.txt", "http://x")


def test_load_nprs_helpers(monkeypatch):
    monkeypatch.setattr(
        datasets.requests, "get", lambda *a, **k: _Resp(text="0.5\n1.5\n")
    )
    assert np.allclose(datasets.load_nprs43(), [0.5, 1.5])
    assert np.allclose(datasets.load_nprs44(), [0.5, 1.5])


# ---------------------------------------------------------------------------
# load_cats
# ---------------------------------------------------------------------------


def _cats_frame():
    idx = pd.date_range("2020-01-01", periods=6, freq="s")
    return pd.DataFrame(
        {"a": np.arange(6.0), "b": np.arange(6.0)[::-1]}, index=idx
    )


def test_load_cats_downloads_and_caches(monkeypatch):
    parquet_bytes = io.BytesIO()
    _cats_frame().to_parquet(parquet_bytes)
    monkeypatch.setattr(
        datasets.requests,
        "get",
        lambda *a, **k: _Resp(
            content=parquet_bytes.getvalue(),
            headers={"content-length": str(len(parquet_bytes.getvalue()))},
        ),
    )
    df = datasets.load_cats()
    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.columns) == ["a", "b"]


def test_load_cats_reads_cache_and_resamples(tmp_path):
    from pathlib import Path

    Path("data/cats").mkdir(parents=True)
    _cats_frame().to_csv("data/cats/data.csv")
    df = datasets.load_cats(resample_s=2)
    assert isinstance(df.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# load_skab
# ---------------------------------------------------------------------------


def test_load_skab_downloads_then_reads(monkeypatch):
    csv_text = "datetime;value;anomaly\n2020-01-01;1.0;0\n2020-01-02;2.0;1\n"

    def fake_get(url, *a, **k):
        if "api.github.com" in url:
            return _Resp(
                json_data=[
                    {
                        "type": "file",
                        "name": "exp.csv",
                        "download_url": "http://x/exp.csv",
                    },
                    {
                        "type": "file",
                        "name": "skip.txt",
                        "download_url": "http://x/skip.txt",
                    },
                ]
            )
        return _Resp(content=csv_text.encode())

    monkeypatch.setattr(datasets.requests, "get", fake_get)
    data = datasets.load_skab()
    from pathlib import Path

    # The .csv was downloaded (.txt skipped); top-level files aren't indexed.
    assert Path("data/skab/exp.csv").exists()
    assert not Path("data/skab/skip.txt").exists()
    assert isinstance(data, dict)


def test_load_skab_reads_existing_cache():
    from pathlib import Path

    sub = Path("data/skab/valve1")
    sub.mkdir(parents=True)
    pd.DataFrame({"v": [1.0, 2.0]}).to_csv(sub / "a.csv", sep=";")
    (sub / "notes.txt").write_text("ignored")  # non-csv skipped
    data = datasets.load_skab()  # dir exists -> no download
    assert "valve1" in data
    assert len(data["valve1"]) == 1


def test_load_skab_api_non_200(monkeypatch):
    monkeypatch.setattr(
        datasets.requests, "get", lambda *a, **k: _Resp(status=500)
    )
    data = datasets.load_skab()  # API error -> nothing downloaded, empty dict
    assert data == {}


def test_load_skab_nested_dir(monkeypatch):
    csv_text = "datetime;value\n2020-01-01;1.0\n"
    calls = {"n": 0}

    def fake_get(url, *a, **k):
        if "api.github.com" in url and calls["n"] == 0:
            calls["n"] += 1
            return _Resp(
                json_data=[
                    {
                        "type": "dir",
                        "name": "valve1",
                        "url": "http://api/valve1",
                    }
                ]
            )
        if "valve1" in url:
            return _Resp(
                json_data=[
                    {
                        "type": "file",
                        "name": "a.csv",
                        "download_url": "http://x/a.csv",
                    }
                ]
            )
        return _Resp(content=csv_text.encode())

    monkeypatch.setattr(datasets.requests, "get", fake_get)
    data = datasets.load_skab()
    assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# load_usp
# ---------------------------------------------------------------------------


def test_load_usp_missing_dir_raises(monkeypatch):
    monkeypatch.setattr(
        datasets.requests, "get", lambda *a, **k: _Resp(status=200)
    )
    with pytest.raises(NotImplementedError, match="download the data"):
        datasets.load_usp()


def _write_usp_arffs(base):
    # Each dataset stresses a different class-handling branch in load_usp.
    arffs = {
        # nominal string labels -> dropped by select_dtypes, encoded as codes
        "chess": "@relation chess\n@attribute f1 numeric\n@attribute outcome {win,lose}\n@data\n1.0,win\n2.0,lose\n",
        # numeric class -> kept by select_dtypes (class already present branch)
        "airlines": "@relation airlines\n@attribute f1 numeric\n@attribute Delay numeric\n@data\n1.0,0\n2.0,1\n",
        # numeric-looking nominal labels -> dropped, but ground truth is numeric
        "gassensor": "@relation gassensor\n@attribute f1 numeric\n@attribute Class {1,2}\n@data\n1.0,1\n2.0,2\n",
        "ozone": "@relation ozone\n@attribute f1 numeric\n@attribute Class {win,lose}\n@data\n1.0,win\n2.0,lose\n",
    }
    for name, text in arffs.items():
        (base / f"{name}.arff").write_text(text)


def test_load_usp_reads_arff():
    from pathlib import Path

    base = Path("data/usp-stream-data/sub")
    base.mkdir(parents=True)
    _write_usp_arffs(base)
    (base / "readme.md").write_text("not an arff")  # non-arff skipped
    data = datasets.load_usp()
    assert "class" in data["chess"].columns  # string labels -> codes
    assert "class" in data["airlines"].columns  # numeric class kept
    assert "class" in data["gassensor"].columns  # numeric ground truth


def test_load_usp_missing_dir_non_200_then_keyerror(monkeypatch):
    # Dir missing + non-200: no raise, walk finds nothing, chess lookup fails.
    monkeypatch.setattr(
        datasets.requests, "get", lambda *a, **k: _Resp(status=503)
    )
    with pytest.raises(KeyError):
        datasets.load_usp()


def test_load_usp_root_dot(monkeypatch):
    # file_path="." makes os.walk yield root == "." (skipped), then KeyError.
    with pytest.raises(KeyError):
        datasets.load_usp(file_path=".")


# ---------------------------------------------------------------------------
# load_bess
# ---------------------------------------------------------------------------


def test_load_bess_downloads(monkeypatch):
    x_csv = "datetime,v\n2020-01-01,1.0\n2020-01-02,2.0\n"
    y_csv = "datetime,label\n2020-01-01,0\n2020-01-02,1\n"

    def fake_get(url, *a, **k):
        return _Resp(
            content=(y_csv if "ground_truth" in url else x_csv).encode()
        )

    monkeypatch.setattr(datasets.requests, "get", fake_get)
    X, y = datasets.load_bess()
    assert isinstance(X.index, pd.DatetimeIndex)
    assert isinstance(y.index, pd.DatetimeIndex)


def test_load_bess_reads_existing_cache(monkeypatch):
    from pathlib import Path

    Path("data/kokam").mkdir(parents=True)
    pd.DataFrame({"v": [1.0]}, index=pd.to_datetime(["2020-01-01"])).to_csv(
        "data/kokam/kokam_norm.csv"
    )
    pd.DataFrame({"label": [0]}, index=pd.to_datetime(["2020-01-01"])).to_csv(
        "data/kokam/kokam_ground_truth.csv"
    )

    def _boom(*a, **k):  # must not be called: files already cached
        raise AssertionError("network used despite cache")

    monkeypatch.setattr(datasets.requests, "get", _boom)
    X, y = datasets.load_bess()
    assert len(X) == 1 and len(y) == 1


def test_load_bess_download_non_200(monkeypatch):
    # Non-200 writes nothing; the subsequent read of a missing file errors.
    monkeypatch.setattr(
        datasets.requests, "get", lambda *a, **k: _Resp(status=404)
    )
    with pytest.raises(FileNotFoundError):
        datasets.load_bess()
