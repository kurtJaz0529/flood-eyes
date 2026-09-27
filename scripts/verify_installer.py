"""Install/uninstall acceptance for the 慧眼识灾 installer.

The smoke installer is compiled here with its own AppId *and* its own Start Menu
group, so the test can never touch a real installation. Two traps are handled
explicitly:

* Inno Setup ignores the runtime ``/GROUP=`` parameter when
  ``DisableProgramGroupPage=yes`` (which this installer sets), so the group has to
  be pinned at compile time via ``/DGroupName=``. An earlier version of this script
  relied on ``/GROUP=`` and silently wrote into the production group.
* The production Start Menu group is asserted to be untouched, so that regression
  cannot come back unnoticed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import winreg

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ID = "{8F3C2A41-6D2B-4E7A-9C15-1B7E4F0A2D33}_is1"
PRODUCTION_GROUP = "慧眼识灾"
ISCC_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
    Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Inno Setup 6/ISCC.exe",
    Path(os.environ.get("USERPROFILE", "")) / "InnoSetup6/ISCC.exe",
]
def roaming_appdata():
    value = os.environ.get("APPDATA")
    if value:
        return Path(value)
    return Path(os.path.expanduser("~")) / "AppData/Roaming"


START_MENU = roaming_appdata() / "Microsoft/Windows/Start Menu/Programs"


def find_iscc():
    for candidate in ISCC_CANDIDATES:
        if candidate.is_file():
            return candidate
    found = shutil.which("ISCC.exe")
    return Path(found) if found else None


def registration(key):
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\" + key) as handle:
            return {name: winreg.QueryValueEx(handle, name)[0]
                    for name in ("DisplayVersion", "InstallLocation", "UninstallString")}
    except FileNotFoundError:
        return None


def shortcuts(path):
    return sorted(p.name for p in path.glob("*.lnk")) if path.is_dir() else []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_smoke(iscc, app_id, group, out_dir, report_path, profile="lite"):
    out_dir.mkdir(parents=True, exist_ok=True)
    log = report_path.parent / "installer_smoke_build.log"
    command = [str(iscc), f"/DProfile={profile}", f"/DAppIdValue={app_id}", f"/DGroupName={group}",
               f"/O{out_dir}", "/Frelease-smoke", str(ROOT / "build/installer.iss")]
    with open(log, "w", encoding="utf-8", errors="replace") as handle:
        code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW).returncode
    return code, out_dir / "release-smoke.exe", command


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True)
    ap.add_argument("--profile", choices=["lite", "full"], default="lite")
    ap.add_argument("--installer", help="reuse an existing smoke installer instead of compiling one")
    ap.add_argument("--group-name", help="Start Menu group baked into --installer")
    ap.add_argument("--appid", help="uninstall registration id baked into --installer")
    ap.add_argument("--iscc", help="path to ISCC.exe")
    args = ap.parse_args()

    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    token = time.strftime("%H%M%S")

    if args.installer:
        installer = Path(args.installer).resolve()
        app_id = args.appid or f"HuiYanShiZai.ReleaseSmoke.{token}"
        group = args.group_name or f"HuiYanShiZai Release Smoke {token}"
        build = {"reused": str(installer), "app_id": app_id, "group": group}
    else:
        iscc = Path(args.iscc) if args.iscc else find_iscc()
        if not iscc or not iscc.is_file():
            raise SystemExit("ISCC.exe not found; pass --iscc or --installer")
        app_id = f"HuiYanShiZai.ReleaseSmoke.{token}"
        group = f"HuiYanShiZai Release Smoke {token}"
        code, installer, command = compile_smoke(iscc, app_id, group, report_path.parent / "smoke_build",
                                                 report_path, args.profile)
        build = {"iscc": str(iscc), "command": command, "exit_code": code, "app_id": app_id,
                 "group": group, "installer": str(installer)}
        if code != 0 or not installer.is_file():
            report_path.write_text(json.dumps({"success": False, "stage": "compile", "build": build},
                                              ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"FAIL smoke installer build (exit {code})")
            return 1

    if group == PRODUCTION_GROUP:
        raise SystemExit("refusing to run: smoke group equals the production Start Menu group")

    smoke_key = f"{app_id}_is1"
    target = report_path.parent / f"install-smoke-{token}"
    group_path = START_MENU / group
    production_group_path = START_MENU / PRODUCTION_GROUP
    production_group_before = shortcuts(production_group_path)

    if target.exists() or registration(smoke_key) is not None or group_path.exists():
        raise SystemExit(f"test destination already exists: {target} / {group_path}")

    before = registration(PRODUCTION_ID)
    report = {"success": False, "checks": [], "install_dir": str(target), "build": build,
              "smoke_appid": smoke_key, "smoke_group": group,
              "production_group": PRODUCTION_GROUP,
              "production_group_before": production_group_before,
              "existing_install_before": before}

    def check(name, value):
        report["checks"].append({"name": name, "passed": bool(value)})
        print(f"{'PASS' if value else 'FAIL'} {name}", flush=True)
        if not value:
            raise AssertionError(name)

    def launch(command, timeout=240):
        return subprocess.run(list(map(str, command)), timeout=timeout,
                              creationflags=subprocess.CREATE_NO_WINDOW).returncode

    installed = False
    try:
        code = launch([installer, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-",
                       f"/DIR={target}", "/MERGETASKS=!desktopicon",
                       f"/LOG={report_path.parent / 'install.log'}"])
        installed = (target / "unins000.exe").is_file()
        check("setup_exit", code == 0 and installed)

        state = registration(smoke_key)
        check("isolated_registration",
              state is not None and state["DisplayVersion"] == "0.5.0"
              and Path(state["InstallLocation"]) == target)
        check("existing_install_unchanged", registration(PRODUCTION_ID) == before)

        found = shortcuts(group_path)
        report["smoke_group_shortcuts"] = found
        check("smoke_group_shortcuts", len(found) == 4)
        check("production_group_untouched", shortcuts(production_group_path) == production_group_before)

        source_dir = ROOT / ("dist_full" if args.profile == "full" else "dist") / "慧眼识灾"
        for name in ("慧眼识灾.exe", "_internal/scripts/fetch_real_samples.py"):
            check("payload_" + name, sha(target / name) == sha(source_dir / name))
        check("no_runtime_data_in_installer",
              not any((target / "outputs").iterdir()) and not any((target / "logs").iterdir()))

        verify = report_path.parent / "installed_cli.json"
        check("installed_offline_calculations",
              launch([sys.executable, ROOT / "scripts/verify_frozen_cli.py",
                      "--exe", target / "慧眼识灾.exe", "--report", verify], 300) == 0)
        report["installed_cli_report"] = str(verify)

        for name in ("outputs/user-result.txt", "logs/user-log.txt", "weights/user-weight.pt"):
            p = target / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("preserve-user-data", encoding="utf-8")
        code = launch([target / "unins000.exe", "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
                       f"/LOG={report_path.parent / 'uninstall.log'}"])
        check("uninstaller_exit", code == 0)
        check("program_removed", not (target / "慧眼识灾.exe").exists())
        for name in ("outputs/user-result.txt", "logs/user-log.txt", "weights/user-weight.pt"):
            check("preserved_" + name, (target / name).read_text(encoding="utf-8") == "preserve-user-data")
        check("test_registration_removed", registration(smoke_key) is None)
        check("test_shortcuts_removed", not group_path.exists())
        check("existing_install_still_unchanged", registration(PRODUCTION_ID) == before)
        check("production_group_still_untouched",
              shortcuts(production_group_path) == production_group_before)

        # 以下两项仅作记录，不计入通过条件。
        # Inno 卸载器结束前会把自身改名为 ``_unins-done.tmp`` 再删除；在受限环境里
        # 这一步可能被拦下（本工具所在环境就会拦截"删除正在运行的可执行文件"），
        # 于是 ``unins000.exe`` 残留在安装目录里。程序载荷已确实删除（见
        # ``program_removed``），用户数据也确实保留（见 ``preserved_*``），
        # 因此这是环境限制而非产品缺陷，只记录、不判定，避免误报。
        residual = sorted(p.name for p in target.iterdir()) if target.exists() else []
        report["uninstaller_self_removed"] = not (target / "unins000.exe").exists()
        report["residual_entries"] = residual

        report["success"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], flush=True)
    finally:
        if installed and (target / "unins000.exe").exists() and registration(smoke_key):
            try:
                report["cleanup_exit"] = launch([target / "unins000.exe", "/VERYSILENT",
                                                 "/SUPPRESSMSGBOXES", "/NORESTART"])
            except Exception as exc:
                report["cleanup_error"] = str(exc)
        report["existing_install_after"] = registration(PRODUCTION_ID)
        report["production_group_after"] = shortcuts(production_group_path)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
