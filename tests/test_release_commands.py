"""Release regressions: failed batches must fail automation; diagnose before UI."""
import json
from pathlib import Path
import zipfile

import pytest

from app import desktop
from scripts import _make_release_zip as packaging
from scripts import run_batch
from src.paths import user_root


def test_headless_connection_failure_returns_without_ui(monkeypatch, capsys):
    monkeypatch.setattr(desktop, "setup_logging", lambda: "desktop.log")
    def unavailable():
        raise TimeoutError("test loopback unavailable")
    monkeypatch.setattr(desktop, "check_loopback", unavailable)
    assert desktop.main(["--headless"]) == 1
    assert "127.0.0.1" in capsys.readouterr().out


def test_diagnosis_records_failure_and_preserves_existing_file(monkeypatch, tmp_path):
    def unavailable():
        raise TimeoutError("test loopback unavailable")
    monkeypatch.setattr(desktop, "check_loopback", unavailable)
    target = tmp_path / "diagnosis.json"
    assert desktop.main(["--diagnose", str(target)]) == 1
    payload = target.read_text(encoding="utf-8")
    assert json.loads(payload)["loopback"]["status"] == "failed"
    with pytest.raises(FileExistsError):
        desktop.main(["--diagnose", str(target)])
    assert target.read_text(encoding="utf-8") == payload


def test_data_dir_isolated_from_bundle(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOOD_DATA_DIR", str(tmp_path / "private-run"))
    assert Path(user_root()) == tmp_path / "private-run"
    assert Path(user_root()).is_dir()


def test_batch_processing_failure_sets_exit_code(monkeypatch, tmp_path):
    from app.automation import get_store
    store = get_store(str(tmp_path))
    store.enqueue({"request": {"lon": 116.3, "lat": 29.15,
        "pre_start": "2020-01-01", "pre_end": "2020-01-03",
        "post_start": "2020-02-01", "post_end": "2020-02-03"},
        "local_pre": str(tmp_path / "missing_pre.tif"),
        "local_post": str(tmp_path / "missing_post.tif")})
    assert run_batch.main(["--out-dir", str(tmp_path), "--run"]) == 1
    assert store.list_jobs()[0]["status"] == "failed"


def test_release_archive_excludes_runtime_data_and_replaces_atomically(tmp_path, monkeypatch):
    source = tmp_path / "慧眼识灾"
    source.mkdir()
    (source / "慧眼识灾.exe").write_bytes(b"binary-fixture")
    for name in ("outputs/private.tif", "logs/desktop.log", "_internal/__pycache__/x.pyc",
                 "_internal/valid.dat"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    target = tmp_path / "release.zip"
    packaging.build_zip(str(source), str(target))
    with zipfile.ZipFile(target) as archive:
        assert set(archive.namelist()) == {"慧眼识灾/慧眼识灾.exe", "慧眼识灾/_internal/valid.dat"}
    previous = target.read_bytes()
    def failed_write(*args, **kwargs):
        raise OSError("simulated full disk")
    monkeypatch.setattr(zipfile.ZipFile, "write", failed_write)
    with pytest.raises(OSError):
        packaging.build_zip(str(source), str(target))
    assert target.read_bytes() == previous
    with pytest.raises(ValueError):
        packaging.build_zip(str(source), str(source / "invalid.zip"))
