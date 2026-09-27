"""HTTPRangeReader 的纯模拟测试。

不联网：用假的 HTTP 响应对象驱动 reader，覆盖跨块读、向后 seek、SEEK_END、
EOF、非 206、Content-Range 错位/不合法、响应截断/超长、LRU 缓存与 read(-1) 上限。

运行：
    python -m pytest tests/test_http_range.py -q
"""

from __future__ import annotations

import io
import os
import re
import sys
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.http_range import (  # noqa: E402
    BLOCK_SIZE,
    MAX_READ_ALL,
    HTTPRangeError,
    HTTPRangeReader,
)

URL = "http://example.invalid/scene.tif"
BLOCK = 64
_RANGE_RE = re.compile(r"^bytes=(\d+)-(\d+)$")


def payload(n: int) -> bytes:
    """确定性、非平凡的字节序列，避免全零掩盖错位/越界。"""
    return bytes((i * 37 + 11) % 256 for i in range(n))


def parse_range(header: str | None) -> tuple[int, int]:
    if header is None:
        raise AssertionError("请求缺少 Range 头（本模块不应发 HEAD 或整文件 GET）")
    match = _RANGE_RE.match(header)
    if match is None:
        raise AssertionError(f"Range 头格式异常: {header!r}")
    return int(match.group(1)), int(match.group(2))


class FakeResponse:
    """最小 HTTP 响应替身，只实现 HTTPRangeReader 用到的部分。"""

    def __init__(self, status: int, headers: dict, body: bytes, chunk: int | None = None):
        self.status = status
        self.headers = headers
        self._body = body
        self._chunk = chunk
        self._pos = 0

    def getheader(self, name: str, default=None):
        for key, value in self.headers.items():
            if key.lower() == name.lower():
                return value
        return default

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._body) - self._pos
        if self._chunk:
            size = min(size, self._chunk)
        chunk = self._body[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


class FakeServer:
    """按 HTTP Range 语义切分 payload，并记录每次请求的 (method, range)。"""

    def __init__(
        self,
        data: bytes,
        *,
        status: int | None = None,
        content_range: str | None = None,
        drop_content_range: bool = False,
        truncate: int = 0,
        extra: bytes = b"",
        chunk: int | None = None,
    ) -> None:
        self.data = data
        self.total = len(data)
        self.status = status
        self.content_range = content_range
        self.drop_content_range = drop_content_range
        self.truncate = truncate
        self.extra = extra
        self.chunk = chunk
        self.requests = []  # 每次请求的 (method, range_header)

    def __call__(self, request: urllib.request.Request) -> FakeResponse:
        method = request.get_method()
        header = request.get_header("Range")
        self.requests.append((method, header))
        start, end = parse_range(header)
        if self.status is not None:
            return FakeResponse(self.status, {}, b"")
        real_end = min(end, self.total - 1)
        body = self.data[start : real_end + 1] + self.extra
        if self.truncate:
            body = body[: -self.truncate]
        headers = {"Content-Length": str(len(body))}
        if not self.drop_content_range:
            headers["Content-Range"] = (
                self.content_range or f"bytes {start}-{real_end}/{self.total}"
            )
        return FakeResponse(206, headers, body, chunk=self.chunk)


def make_reader(data: bytes, *, block_size: int = BLOCK, max_blocks: int = 32, **server_kw):
    server = FakeServer(data, **server_kw)
    reader = HTTPRangeReader(URL, server, block_size=block_size, max_blocks=max_blocks)
    return reader, server


# ----------------------------------------------------------------------
# 首个请求 / 总长度
# ----------------------------------------------------------------------


def test_first_request_is_get_range_zero_to_block_minus_one():
    data = payload(500)
    server = FakeServer(data)
    reader = HTTPRangeReader(URL, server)  # 默认块大小
    assert reader.read(4) == data[:4]
    assert server.requests == [("GET", f"bytes=0-{BLOCK_SIZE - 1}")]
    assert reader.size == len(data)


def test_open_request_receives_urllib_request():
    seen = []

    def opener(request):
        seen.append(request)
        start, end = parse_range(request.get_header("Range"))
        return FakeResponse(
            206,
            {"Content-Range": f"bytes {start}-{end}/{end + 1}"},
            b"x" * (end - start + 1),
        )

    reader = HTTPRangeReader(URL, opener, block_size=8)
    reader.read(4)
    assert isinstance(seen[0], urllib.request.Request)
    assert seen[0].full_url == URL
    assert "identity" in seen[0].headers.values(), "Range 读取必须要求不做内容编码"


def test_seek_end_before_any_read_fetches_only_first_block():
    reader, server = make_reader(payload(200))
    assert reader.seek(0, io.SEEK_END) == 200
    assert reader.tell() == 200
    assert len(server.requests) == 1


def test_only_get_with_range_headers_are_issued():
    reader, server = make_reader(payload(500))
    reader.read(500)
    assert server.requests
    assert all(method == "GET" for method, _ in server.requests), "不得请求 HEAD"
    assert all(header and header.startswith("bytes=") for _, header in server.requests)


# ----------------------------------------------------------------------
# 跨块 / 向后 seek / SEEK_END / EOF
# ----------------------------------------------------------------------


def test_read_crosses_block_boundaries():
    data = payload(300)
    reader, server = make_reader(data)
    reader.seek(40)
    assert reader.read(100) == data[40:140]
    assert reader.tell() == 140
    assert [header for _, header in server.requests] == [
        "bytes=0-63",  # _ensure_total 先取第一块拿总长度
        "bytes=64-127",
        "bytes=128-191",
    ]


def test_read_from_mid_block_offset():
    data = payload(300)
    reader, _ = make_reader(data)
    reader.seek(0, io.SEEK_END)
    reader.seek(70)
    assert reader.read(30) == data[70:100]


def test_backward_seek_is_served_from_cache():
    data = payload(300)
    reader, server = make_reader(data)
    assert reader.read(50) == data[:50]
    before = len(server.requests)
    assert reader.seek(0) == 0
    assert reader.read(50) == data[:50]
    assert len(server.requests) == before, "回退重读同一块不应再发请求"


def test_seek_end_reads_tail():
    data = payload(200)
    reader, _ = make_reader(data)
    assert reader.seek(0, io.SEEK_END) == len(data)
    assert reader.read(8) == b""
    assert reader.seek(-6, io.SEEK_END) == len(data) - 6
    assert reader.read(6) == data[-6:]


def test_read_all_matches_payload_across_many_blocks():
    data = payload(1000)
    reader, _ = make_reader(data)
    assert reader.read() == data
    assert reader.read() == b"", "EOF 处 read 返回空"


def test_read_zero_returns_empty_without_request():
    reader, server = make_reader(payload(300))
    assert reader.read(0) == b""
    assert server.requests == []


def test_seek_beyond_eof_then_read_is_empty():
    reader, _ = make_reader(payload(100))
    reader.seek(0, io.SEEK_END)
    reader.seek(1000)
    assert reader.read(10) == b""


def test_seek_past_eof_before_total_is_known_reads_empty():
    reader, server = make_reader(payload(100))
    reader.seek(1000)  # 此时总长度未知
    assert reader.read(10) == b""
    assert len(server.requests) == 1, "只应发一次拿总长度的请求"


def test_seek_negative_targets_are_rejected():
    reader, _ = make_reader(payload(200))
    with pytest.raises(ValueError):
        reader.seek(-1, io.SEEK_SET)
    with pytest.raises(ValueError):
        reader.seek(-1, io.SEEK_CUR)
    with pytest.raises(ValueError):
        reader.seek(-201, io.SEEK_END)  # 200 - 201 = -1
    assert reader.tell() == 0, "被拒绝的 seek 不得改动位置"


def test_seek_unknown_whence_is_rejected():
    reader, _ = make_reader(payload(200))
    with pytest.raises(ValueError):
        reader.seek(0, 7)


# ----------------------------------------------------------------------
# readinto / tell / 能力位 / close
# ----------------------------------------------------------------------


def test_readinto_fills_buffer_and_advances():
    data = payload(300)
    reader, _ = make_reader(data)
    buf = bytearray(100)
    assert reader.readinto(buf) == 100
    assert bytes(buf) == data[:100]
    assert reader.tell() == 100

    reader.seek(50)
    small = bytearray(80)
    assert reader.readinto(small) == 80
    assert bytes(small) == data[50:130]


def test_readinto_at_eof_returns_zero():
    reader, _ = make_reader(payload(100))
    reader.seek(0, io.SEEK_END)
    buf = bytearray(10)
    assert reader.readinto(buf) == 0
    assert reader.readinto(bytearray()) == 0


def test_capabilities_and_tell():
    reader, _ = make_reader(payload(10))
    assert reader.readable() is True
    assert reader.seekable() is True
    assert reader.writable() is False
    assert reader.tell() == 0
    assert reader.seek(5) == 5
    assert reader.tell() == 5


def test_close_is_terminal_and_idempotent():
    reader, _ = make_reader(payload(300))
    reader.read(10)
    reader.close()
    assert reader.closed is True
    assert reader.readable() is False
    assert reader.seekable() is False
    with pytest.raises(ValueError):
        reader.read(1)
    with pytest.raises(ValueError):
        reader.seek(0)
    with pytest.raises(ValueError):
        reader.tell()
    reader.close()  # 幂等


def test_read_rejects_float_size():
    reader, _ = make_reader(payload(300))
    with pytest.raises(TypeError):
        reader.read(1.5)


# ----------------------------------------------------------------------
# 协议校验：非 206 / Content-Range / 截断 / 超长
# ----------------------------------------------------------------------


def test_status_200_is_rejected():
    reader, _ = make_reader(payload(200), status=200)
    with pytest.raises(HTTPRangeError, match="206"):
        reader.read(1)


def test_other_non_206_status_is_rejected():
    reader, _ = make_reader(payload(200), status=416)
    with pytest.raises(HTTPRangeError):
        reader.read(1)


def test_missing_content_range_is_rejected():
    reader, _ = make_reader(payload(300), drop_content_range=True)
    with pytest.raises(HTTPRangeError, match="Content-Range"):
        reader.read(1)


def test_misaligned_content_range_is_rejected():
    reader, _ = make_reader(payload(300), content_range="bytes 1-64/300")
    with pytest.raises(HTTPRangeError, match="错位"):
        reader.read(1)


def test_illegal_content_range_is_rejected():
    reader, _ = make_reader(payload(300), content_range="bytes 0-63/*")
    with pytest.raises(HTTPRangeError):
        reader.read(1)


def test_content_range_end_beyond_total_is_rejected():
    reader, _ = make_reader(payload(300), content_range="bytes 0-63/40")
    with pytest.raises(HTTPRangeError):
        reader.read(1)


def test_truncated_body_is_rejected():
    reader, _ = make_reader(payload(300), truncate=1)
    with pytest.raises(HTTPRangeError, match="截断"):
        reader.read(1)


def test_oversized_body_is_rejected():
    reader, _ = make_reader(payload(300), extra=b"X" * 4)
    with pytest.raises(HTTPRangeError, match="超出"):
        reader.read(1)


def test_inconsistent_total_across_blocks_is_rejected():
    data = payload(300)
    calls = {"n": 0}

    def opener(request):
        start, end = parse_range(request.get_header("Range"))
        calls["n"] += 1
        total = 300 if calls["n"] == 1 else 299
        real_end = min(end, total - 1)
        return FakeResponse(
            206,
            {"Content-Range": f"bytes {start}-{real_end}/{total}"},
            data[start : real_end + 1],
        )

    reader = HTTPRangeReader(URL, opener, block_size=BLOCK)
    assert reader.read(64) == data[:64]
    with pytest.raises(HTTPRangeError, match="不一致"):
        reader.read(64)


def test_chunked_transport_is_reassembled():
    data = payload(300)
    reader, _ = make_reader(data, chunk=7)
    assert reader.read(100) == data[:100]


# ----------------------------------------------------------------------
# LRU 缓存
# ----------------------------------------------------------------------


def test_lru_cache_evicts_least_recently_used_block():
    reader, server = make_reader(payload(200), block_size=16, max_blocks=2)
    reader.read(16)  # 块 0
    reader.seek(16)
    reader.read(16)  # 块 1
    assert len(server.requests) == 2
    reader.seek(32)
    reader.read(16)  # 块 2 → 淘汰块 0
    assert len(server.requests) == 3
    reader.seek(0)
    reader.read(16)  # 块 0 需重取
    assert len(server.requests) == 4


def test_lru_cache_keeps_recently_touched_block():
    reader, server = make_reader(payload(200), block_size=16, max_blocks=2)
    reader.read(16)  # 块 0
    reader.seek(16)
    reader.read(16)  # 块 1
    reader.seek(0)
    reader.read(16)  # 触摸块 0 → 块 1 变为最久未用
    before = len(server.requests)
    reader.seek(32)
    reader.read(16)  # 块 2 → 淘汰块 1
    reader.seek(16)
    reader.read(16)  # 块 1 需重取
    assert len(server.requests) == before + 2


def test_single_block_cache_still_works():
    data = payload(200)
    reader, server = make_reader(data, block_size=16, max_blocks=1)
    assert reader.read(16) == data[:16]
    reader.seek(32)
    assert reader.read(16) == data[32:48]
    reader.seek(0)
    assert reader.read(16) == data[:16]
    assert len(server.requests) == 3  # 块0（顺带拿总长度）+ 块2 + 重取块0


# ----------------------------------------------------------------------
# read(-1) 上限
# ----------------------------------------------------------------------


def test_read_all_is_capped_at_64_mib():
    total = MAX_READ_ALL + 1
    block = 1024

    def opener(request):
        start, end = parse_range(request.get_header("Range"))
        real_end = min(end, total - 1)
        return FakeResponse(
            206,
            {"Content-Range": f"bytes {start}-{real_end}/{total}"},
            b"\x00" * (real_end - start + 1),
        )

    reader = HTTPRangeReader(URL, opener, block_size=block)
    with pytest.raises(ValueError, match="上限"):
        reader.read()
    assert reader.read(4) == b"\x00" * 4, "显式 size 不受 read(-1) 上限影响"


# ----------------------------------------------------------------------
# 构造参数校验
# ----------------------------------------------------------------------


def test_constructor_validation():
    server = FakeServer(payload(10))
    with pytest.raises(TypeError):
        HTTPRangeReader("", server)
    with pytest.raises(TypeError):
        HTTPRangeReader(URL, None)
    with pytest.raises(ValueError):
        HTTPRangeReader(URL, server, block_size=0)
    with pytest.raises(ValueError):
        HTTPRangeReader(URL, server, max_blocks=0)
