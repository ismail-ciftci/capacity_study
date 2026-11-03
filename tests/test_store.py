import os

import pytest

from capacity_study import store


def _flaky_replace(failures: int):
    real, calls = os.replace, {"n": 0}

    def replace(src, dst):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise PermissionError(13, "Access is denied")
        real(src, dst)
    return replace, calls


def test_atomic_write_survives_a_transient_lock(tmp_path, monkeypatch):
    replace, calls = _flaky_replace(2)
    monkeypatch.setattr(store.os, "replace", replace)
    monkeypatch.setattr(store, "REPLACE_BACKOFF_S", 0.0)
    store.atomic_write_json(tmp_path / "a.json", {"x": 1})
    assert (tmp_path / "a.json").read_text() and calls["n"] == 3
    assert not (tmp_path / "a.json.tmp").exists()


def test_atomic_write_reraises_a_persistent_lock(tmp_path, monkeypatch):
    replace, calls = _flaky_replace(10 ** 6)
    monkeypatch.setattr(store.os, "replace", replace)
    monkeypatch.setattr(store, "REPLACE_BACKOFF_S", 0.0)
    with pytest.raises(PermissionError):
        store.atomic_write_json(tmp_path / "a.json", {"x": 1})
    assert calls["n"] == store.REPLACE_ATTEMPTS
