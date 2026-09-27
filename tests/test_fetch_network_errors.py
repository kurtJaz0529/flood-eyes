"""在线影像读取失败时的归因与显式降级开关。

背景（2026-09-26 实测，证据见 outputs/release_v0.5.0/online_pipeline.json 与 tls_probe.json）：
本机通过 /vsicurl/ 读取 AWS Sentinel-2 COG 时，每一景都报
``RasterioIOError: CURL error: schannel: the revocation status is unknown``，
即 schannel 无法完成证书吊销查询，导致在线链路整体失败。
GDAL 3.12 未暴露"只关吊销检查"的配置项（DLL 内只有全关的 GDAL_HTTP_UNSAFESSL），
所以产品侧只提供显式的、默认关闭的开关，并把结论翻译成可执行的说明。
"""

import pytest

from scripts import fetch_real_samples as fetch

REVOCATION = ("RasterioIOError: CURL error: schannel: "
              "the revocation status is unknown")


def test_revocation_error_is_explained_with_opt_in(monkeypatch):
    hint = fetch.describe_network_error(RuntimeError(REVOCATION))
    assert "吊销" in hint
    assert "FLOOD_ALLOW_UNSAFE_TLS=1" in hint, "必须告诉用户确切的开关名"
    assert "tls_verification_disabled" in hint, "必须说明会记录降级状态"


def test_timeout_and_dns_are_distinguished(monkeypatch):
    timeout = fetch.describe_network_error(RuntimeError("CURL error: Connection timed out after 10008 milliseconds"))
    assert "超时" in timeout and "FLOOD_ALLOW_UNSAFE_TLS" not in timeout

    dns = fetch.describe_network_error(RuntimeError("Could not resolve host: example.invalid"))
    assert "解析" in dns


def test_unrelated_error_yields_no_hint():
    assert fetch.describe_network_error(RuntimeError("band index out of range")) == ""


# ---------------------------------------------------------------------------
# 本地化文案：Windows 系统错误信息会跟随系统语言，只有 WinError 数字码是稳定的。
# 本机（中文系统）实测：STAC 检索失败抛的是
# URLError: <urlopen error [WinError 10060] 由于连接方在一段时间后没有正确答复…>
# 只匹配英文 "timed out" 会全部漏判。
# ---------------------------------------------------------------------------

LOCALIZED_TIMEOUT = ("URLError: <urlopen error [WinError 10060] "
                     "由于连接方在一段时间后没有正确答复或连接的主机没有反应，连接尝试失败。>")
LOCALIZED_REFUSED = ("URLError: <urlopen error [WinError 10061] "
                     "由于目标计算机积极拒绝，无法连接。>")


def test_localized_winerror_timeout_is_still_classified():
    hint = fetch.describe_network_error(RuntimeError(LOCALIZED_TIMEOUT))
    assert "超时" in hint, f"中文系统文案必须仍能判为超时，实际={hint!r}"


def test_localized_winerror_refused_is_classified():
    hint = fetch.describe_network_error(RuntimeError(LOCALIZED_REFUSED))
    assert "拒绝" in hint


def test_winerror_dns_is_classified():
    hint = fetch.describe_network_error(RuntimeError("[WinError 11001] 找不到主机"))
    assert "解析" in hint


def test_exception_types_alone_are_enough():
    assert "超时" in fetch.describe_network_error(TimeoutError("timed out"))
    assert "拒绝" in fetch.describe_network_error(ConnectionRefusedError(10061, "refused"))


def test_urlerror_reason_is_inspected():
    """urllib 把底层异常包在 .reason 里，类型信息不能丢。"""
    import urllib.error

    wrapped = urllib.error.URLError(TimeoutError("timed out"))
    assert "超时" in fetch.describe_network_error(wrapped)


def test_unsafe_tls_hint_is_not_offered_for_plain_timeouts():
    hint = fetch.describe_network_error(RuntimeError(LOCALIZED_TIMEOUT))
    assert "FLOOD_ALLOW_UNSAFE_TLS" not in hint, "纯网络不通不该建议关闭证书校验"


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("YES", True), ("On", True),
    ("0", False), ("", False), ("no", False),
])
def test_unsafe_tls_is_opt_in_only(monkeypatch, value, expected):
    monkeypatch.setenv("FLOOD_ALLOW_UNSAFE_TLS", value)
    assert fetch.allow_unsafe_tls() is expected


def test_gdal_env_keeps_verification_by_default(monkeypatch):
    monkeypatch.delenv("FLOOD_ALLOW_UNSAFE_TLS", raising=False)
    env = fetch._gdal_env()
    assert "GDAL_HTTP_UNSAFESSL" not in env, "默认绝不能关闭证书校验"


def test_gdal_env_honours_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("FLOOD_ALLOW_UNSAFE_TLS", "1")
    env = fetch._gdal_env()
    assert env["GDAL_HTTP_UNSAFESSL"] == "YES"


def test_deadline_still_overrides_timeout_when_opt_in(monkeypatch):
    """降级开关不能挤掉 deadline 收紧后的超时值。"""
    import time as _time

    monkeypatch.setenv("FLOOD_ALLOW_UNSAFE_TLS", "1")
    monkeypatch.setattr(fetch, "_DEADLINE", _time.time() + 3)
    env = fetch._gdal_env()
    assert env["GDAL_HTTP_UNSAFESSL"] == "YES"
    assert env["GDAL_HTTP_TIMEOUT"] == "8"


def test_provenance_records_the_degradation(monkeypatch):
    monkeypatch.delenv("FLOOD_ALLOW_UNSAFE_TLS", raising=False)
    assert fetch.provenance_common()["tls_verification_disabled"] is False

    monkeypatch.setenv("FLOOD_ALLOW_UNSAFE_TLS", "1")
    assert fetch.provenance_common()["tls_verification_disabled"] is True


def test_scene_picking_reports_the_network_cause(monkeypatch, capsys):
    """全部景都因网络失败时，要把根因打印出来，而不是只报"被跳过"。"""
    monkeypatch.setattr(fetch, "_DEADLINE", None)

    def broken(*args, **kwargs):
        raise RuntimeError(REVOCATION)

    monkeypatch.setattr(fetch, "find_valid_center", broken)
    picked = fetch._pick_scene([{"id": "S2B_TEST", "bbox": [115, 28, 117, 30],
                                 "properties": {"datetime": "2020-06-01T00:00:00Z",
                                                "eo:cloud_cover": 5}}],
                               115.5, 28.5, "2020-06-01", 35, 5.0)
    assert picked is None
    out = capsys.readouterr().out
    assert "吊销" in out
    assert "FLOOD_ALLOW_UNSAFE_TLS=1" in out
