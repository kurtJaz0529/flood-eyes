"""桌面启动器回归：app.desktop.bypass_proxy_for_localhost 的代理绕行契约。

覆盖的验收契约：
    * 合并 NO_PROXY 与 no_proxy 中现有的不重复条目；
    * 补齐 127.0.0.1 / localhost / ::1；
    * 结果同步写回两种大小写；
    * 不改动 HTTP_PROXY / HTTPS_PROXY；
    * 连续调用两次幂等；
    * 已有的 "*" 通配条目保留。

只跑本文件：python -m pytest tests/test_desktop_startup.py -q
"""

from __future__ import annotations

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 按文件路径加载，避免 import app 包时触发无关的界面依赖；仅执行模块顶层代码，
# 不会启动服务或线程。
_SPEC = importlib.util.spec_from_file_location(
    "desktop_startup_under_test", os.path.join(ROOT, "app", "desktop.py"))
_DESKTOP = importlib.util.module_from_spec(_SPEC)
sys.modules["desktop_startup_under_test"] = _DESKTOP
_SPEC.loader.exec_module(_DESKTOP)

bypass_proxy_for_localhost = _DESKTOP.bypass_proxy_for_localhost
check_loopback = _DESKTOP.check_loopback
classify_loopback_failure = _DESKTOP.classify_loopback_failure
firewall_remedy = _DESKTOP.firewall_remedy
loopback_failure_message = _DESKTOP.loopback_failure_message
LoopbackError = _DESKTOP.LoopbackError

LOOPBACK = ("127.0.0.1", "localhost", "::1")


def _entries(value):
    """把代理变量拆成条目列表（去空白、去空项），用于比较内容而非顺序。"""
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _clean_proxy_env(monkeypatch):
    """用大小写敏感的字典覆盖跨平台契约（Windows 环境键不区分大小写）。"""
    monkeypatch.setattr(os, "environ", {})


def test_merges_both_cases_adds_loopback_and_syncs(monkeypatch):
    _clean_proxy_env(monkeypatch)
    monkeypatch.setenv("NO_PROXY", "example.com,127.0.0.1")
    monkeypatch.setenv("no_proxy", "example.org")

    bypass_proxy_for_localhost()

    upper = _entries(os.environ["NO_PROXY"])
    lower = _entries(os.environ["no_proxy"])

    assert upper == lower, "NO_PROXY 与 no_proxy 必须同步写回同一条目集"
    assert len(upper) == len(set(upper)), "合并后不允许出现重复条目"
    assert set(upper) >= {"example.com", "example.org"}, "两侧已有条目都必须保留"
    assert set(upper) >= set(LOOPBACK), "必须补齐回环地址"


def test_preserves_http_proxy_values(monkeypatch):
    _clean_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.corp:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp:8443")

    bypass_proxy_for_localhost()

    assert os.environ["HTTP_PROXY"] == "http://proxy.corp:8080"
    assert os.environ["HTTPS_PROXY"] == "http://proxy.corp:8443"


def test_idempotent_when_called_twice(monkeypatch):
    _clean_proxy_env(monkeypatch)
    monkeypatch.setenv("NO_PROXY", "example.com")
    monkeypatch.setenv("no_proxy", "example.org")

    bypass_proxy_for_localhost()
    first = (os.environ["NO_PROXY"], os.environ["no_proxy"])
    bypass_proxy_for_localhost()
    second = (os.environ["NO_PROXY"], os.environ["no_proxy"])

    assert first == second, "重复调用不得改动已写入的值"
    assert len(_entries(second[0])) == len(set(_entries(second[0]))), "重复调用不得堆积重复项"


def test_preserves_wildcard_entry(monkeypatch):
    _clean_proxy_env(monkeypatch)
    monkeypatch.setenv("NO_PROXY", "*")

    bypass_proxy_for_localhost()

    assert "*" in _entries(os.environ["NO_PROXY"]), "通配符 * 必须保留"
    assert "*" in _entries(os.environ["no_proxy"]), "通配符 * 必须同步到 no_proxy"


# ---------------------------------------------------------------------------
# 回环自检失败的归因与处置
#
# 背景：Windows 上 socket.socketpair() 用回环 TCP 对模拟且无超时；入站被过滤时
# asyncio 建事件循环会永久阻塞。已实测（见 outputs/release_v0.5.0/net_compare.json）
# 同一份 python.exe 改名副本同样超时、原名正常，说明过滤按可执行文件身份生效。
# ---------------------------------------------------------------------------


class _FakeSocket:
    """按脚本驱动的最小 socket 替身；fail_at 指定在哪一步抛错。"""

    def __init__(self, fail_at=None, exc=None):
        self.fail_at = fail_at
        self.exc = exc

    def settimeout(self, value):
        pass

    def bind(self, addr):
        if self.fail_at == "bind":
            raise self.exc

    def listen(self, backlog=1):
        if self.fail_at == "listen":
            raise self.exc

    def getsockname(self):
        return ("127.0.0.1", 51234)

    def connect(self, addr):
        if self.fail_at == "connect":
            raise self.exc

    def accept(self):
        if self.fail_at == "accept":
            raise self.exc
        return _FakeSocket(), ("127.0.0.1", 51234)

    def sendall(self, payload):
        pass

    def recv(self, size):
        return b"ok"

    def close(self):
        pass


def _patch_sockets(monkeypatch, fail_at, exc):
    def factory(*args, **kwargs):
        return _FakeSocket(fail_at=fail_at, exc=exc)

    monkeypatch.setattr(_DESKTOP.socket, "socket", factory)


def test_remedy_targets_only_this_exe_with_tcp(monkeypatch):
    remedy = firewall_remedy(r"C:\some dir\慧眼识灾.exe")
    assert "dir=in" in remedy and "action=allow" in remedy
    assert "protocol=TCP" in remedy, "只放行界面所需的 TCP，不放 UDP"
    assert r"C:\some dir\慧眼识灾.exe" in remedy
    assert remedy.startswith("netsh advfirewall firewall add rule")


def test_timeout_is_classified_as_inbound_filtering(monkeypatch):
    error = classify_loopback_failure("connect", TimeoutError("timed out"), r"C:\app\a.exe")
    assert isinstance(error, OSError), "必须继承 OSError，兼容既有 except OSError 调用方"
    assert error.cause == "loopback_timeout"
    assert error.remedy, "入站过滤必须给出可执行的解决办法"
    assert "C:\\app\\a.exe" in error.remedy


def test_refused_is_not_blamed_on_the_firewall(monkeypatch):
    error = classify_loopback_failure("accept", ConnectionRefusedError(10061, "refused"),
                                      r"C:\app\a.exe")
    assert error.cause == "not_listening"
    assert error.remedy == "", "程序自身没在监听时，不该让用户去改防火墙"


def test_check_loopback_reports_the_failing_stage(monkeypatch):
    _patch_sockets(monkeypatch, "connect", TimeoutError("timed out"))
    try:
        check_loopback(timeout=0.01)
    except LoopbackError as exc:
        assert exc.stage == "connect"
        assert exc.cause == "loopback_timeout"
    else:
        raise AssertionError("回环连接超时必须抛出 LoopbackError")


def test_check_loopback_succeeds_when_loopback_works(monkeypatch):
    _patch_sockets(monkeypatch, None, None)
    assert check_loopback(timeout=0.01)["status"] == "ok"


def test_failure_message_carries_cause_and_remedy(monkeypatch):
    error = classify_loopback_failure("connect", TimeoutError("timed out"), r"C:\app\a.exe")
    message = loopback_failure_message(error, r"C:\app\logs\desktop.log")
    assert "无法建立本机连接" in message
    assert "allow_loopback.ps1" in message, "要指出现成的修复脚本"
    assert "netsh advfirewall firewall add rule" in message, "要给出可直接粘贴的命令"
    assert r"C:\app\a.exe" in message
    assert "desktop.log" in message


def test_diagnose_dict_is_serialisable(monkeypatch):
    import json

    error = classify_loopback_failure("connect", TimeoutError("timed out"), r"C:\app\a.exe")
    payload = error.as_dict()
    assert payload["status"] == "failed"
    assert payload["cause"] == "loopback_timeout"
    json.dumps(payload, ensure_ascii=False)


def test_remedy_is_limited_to_loopback():
    remedy = firewall_remedy(r"C:\app\a.exe")
    assert "localip=127.0.0.1" in remedy
    assert "remoteip=127.0.0.1" in remedy


def test_receive_timeout_does_not_recommend_firewall_changes():
    error = classify_loopback_failure("recv", TimeoutError("timed out"), r"C:\app\a.exe")
    assert error.cause == "unknown"
    assert error.remedy == ""
