"""Exercise Windows helper path resolution without touching firewall or UAC."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell.exe")
pytestmark = pytest.mark.skipif(not POWERSHELL, reason="Windows PowerShell required")


@pytest.mark.parametrize("packaged", [False, True])
def test_dry_run_resolves_layout_and_limits_addresses(tmp_path, packaged):
    script = tmp_path / ("_internal/scripts" if packaged else "scripts") / "allow_loopback.ps1"
    exe = tmp_path / ("慧眼识灾.exe" if packaged else "dist/慧眼识灾/慧眼识灾.exe")
    script.parent.mkdir(parents=True)
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.touch()
    shutil.copyfile(ROOT / "scripts/allow_loopback.ps1", script)
    result = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass",
                             "-File", str(script), "-DryRun"], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    # The current Windows session redirects PowerShell output as UTF-8.
    text = result.stdout.decode("utf-8")
    config = json.loads(text)
    assert Path(config["Program"]) == exe
    assert config["LocalAddress"] == config["RemoteAddress"] == "127.0.0.1"
    assert config["Protocol"] == "TCP"


def test_remove_preview_does_not_require_installed_exe(tmp_path):
    script = tmp_path / "allow_loopback.ps1"
    shutil.copyfile(ROOT / "scripts/allow_loopback.ps1", script)
    result = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass",
                             "-File", str(script), "-Remove", "-DryRun"],
                            capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.decode("utf-8"))["Remove"] is True
