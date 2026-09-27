"""基于 HTTP Range 的只读、可 seek 文件对象。

用途
====
给 ``rasterio.open(..., opener=...)`` 提供一个不依赖 GDAL ``/vsicurl/`` 的
读取通道。本仓库实测 vsicurl 在 Windows/schannel 下会因证书吊销检查失败，
且 Range 响应被截断时错误归因不清（见 tests/test_fetch_network_errors.py）。
本模块把"按块取 Range、校验 Content-Range、LRU 缓存"独立出来，便于测试与替换。

设计边界
========
* URL 安全校验（协议/主机/IP）、代理、HTTPS 证书校验、超时与重试，全部由调用方
  提供的 ``open_request`` 负责。本类**不**关闭 SSL 校验、不换用其他网络客户端、
  不重试、不请求 HEAD。
* ``open_request`` 接收 ``urllib.request.Request``，返回一个 context manager，
  其 ``__enter__`` 结果为 HTTP 响应对象（需提供 ``status`` 或 ``getcode()``、
  ``getheader()`` 或 ``headers``、``read()``）。``urllib.request.urlopen`` 的返回值
  直接满足该契约。
* 懒加载：首次读取才发第一个请求 ``Range: bytes=0-<block_size-1>``，
  从 206 的 ``Content-Range: bytes START-END/TOTAL`` 得到总长度。
* 只接受 206。200（服务器忽略 Range，会把整景灌进来）、Content-Range 错位或
  不合法、响应被截断、响应超出 Content-Range 声明长度，一律报错。
* 每次请求最多读 ``expected + 1`` 字节：既能发现"服务器多给了"，又不会被超大
  响应拖垮内存。
* ``read(-1)`` 最多返回 64 MiB，超过即报错，避免误把整景读进内存。

用法
====
::

    import urllib.request
    import rasterio
    from src.http_range import HTTPRangeReader

    def open_request(req):
        # 这里做 URL 白名单、代理、证书校验与超时控制。
        return urllib.request.urlopen(req, timeout=30)

    with rasterio.open(url, opener=lambda u: HTTPRangeReader(u, open_request)) as src:
        ...
"""

from __future__ import annotations

import io
import operator
import re
import urllib.request
from collections import OrderedDict
from typing import Any, Callable, Optional

__all__ = [
    "HTTPRangeReader",
    "HTTPRangeError",
    "BLOCK_SIZE",
    "MAX_BLOCKS",
    "MAX_READ_ALL",
]

BLOCK_SIZE = 256 * 1024  # 262144：首个请求即 bytes=0-262143
MAX_BLOCKS = 32  # 缓存上限 32 块 ≈ 8 MiB
MAX_READ_ALL = 64 * 1024 * 1024  # read(-1) 的硬上限

_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$")


class HTTPRangeError(OSError):
    """服务器未按 HTTP Range 语义响应，或响应被截断/超长。"""


class HTTPRangeReader(io.RawIOBase):
    """把支持 Range 的 HTTP(S) 资源当作可 seek 的只读二进制流。

    参数
    ----
    url
        资源地址，原样交给 ``open_request``；本类不做协议/主机校验。
    open_request
        可调用对象，接收 ``urllib.request.Request``，返回 HTTP 响应的 context manager。
    block_size
        单次 Range 请求的字节数，默认 256 KiB。
    max_blocks
        LRU 块缓存上限，默认 32 块。
    """

    def __init__(
        self,
        url: str,
        open_request: Callable[[urllib.request.Request], Any],
        block_size: int = BLOCK_SIZE,
        max_blocks: int = MAX_BLOCKS,
    ) -> None:
        super().__init__()
        if not isinstance(url, str) or not url:
            raise TypeError("url 必须是非空字符串")
        if not callable(open_request):
            raise TypeError("open_request 必须是可调用对象")
        if not _is_positive_int(block_size):
            raise ValueError("block_size 必须是正整数")
        if not _is_positive_int(max_blocks):
            raise ValueError("max_blocks 必须是正整数")

        self._url = url
        self._open_request = open_request
        self._block_size = block_size
        self._max_blocks = max_blocks
        self._pos = 0
        self._total: Optional[int] = None
        self._cache: "OrderedDict[int, bytes]" = OrderedDict()

    # ------------------------------------------------------------------
    # 只读元信息
    # ------------------------------------------------------------------

    @property
    def url(self) -> str:
        return self._url

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def max_blocks(self) -> int:
        return self._max_blocks

    @property
    def size(self) -> Optional[int]:
        """资源总字节数；首次读取（或 SEEK_END）之前为 ``None``。"""
        return self._total

    # ------------------------------------------------------------------
    # io 接口
    # ------------------------------------------------------------------

    def readable(self) -> bool:
        return not self.closed

    def seekable(self) -> bool:
        return not self.closed

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        self._check_closed()
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """支持 SET/CUR/END；计算后的目标为负则拒绝，越过 EOF 则允许。"""
        self._check_closed()
        offset = operator.index(offset)
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self._pos + offset
        elif whence == io.SEEK_END:
            target = self._ensure_total() + offset
        else:
            raise ValueError(f"不支持的 whence: {whence!r}")
        if target < 0:
            raise ValueError(f"seek 目标为负: {target}")
        self._pos = target
        return target

    def read(self, size: int = -1) -> bytes:
        self._check_closed()
        if size is None:
            size = -1
        else:
            size = operator.index(size)
        if size < 0:
            remaining = self._ensure_total() - self._pos
            if remaining > MAX_READ_ALL:
                raise ValueError(
                    f"read(-1) 需要 {remaining} 字节，超过上限 {MAX_READ_ALL}；"
                    "请改用窗口/分块读取，避免把整景读入内存"
                )
            size = remaining
        if size == 0:
            return b""
        data = self._gather(self._pos, size)
        self._pos += len(data)
        return data

    def readinto(self, b: Any) -> int:
        self._check_closed()
        view = memoryview(b).cast("B")
        if len(view) == 0:
            return 0
        data = self._gather(self._pos, len(view))
        view[: len(data)] = data
        self._pos += len(data)
        return len(data)

    def close(self) -> None:
        cache = getattr(self, "_cache", None)
        if cache is not None:
            cache.clear()
        super().close()

    # ------------------------------------------------------------------
    # 内部：块缓存与 Range 请求
    # ------------------------------------------------------------------

    def _check_closed(self) -> None:
        if self.closed:
            raise ValueError("I/O operation on closed HTTPRangeReader")

    def _ensure_total(self) -> int:
        """保证总长度已知（必要时取第一块）。"""
        if self._total is None:
            self._get_block(0)
        total = self._total
        if total is None:
            raise HTTPRangeError("无法确定资源总长度")
        return total

    def _get_block(self, index: int) -> bytes:
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        block = self._fetch_block(index)
        if block:
            if len(self._cache) >= self._max_blocks:
                self._cache.popitem(last=False)
            self._cache[index] = block
        return block

    def _fetch_block(self, index: int) -> bytes:
        start = index * self._block_size
        if self._total is not None:
            if start >= self._total:
                return b""
            end = min(start + self._block_size - 1, self._total - 1)
        else:
            end = start + self._block_size - 1
        return self._request_range(start, end)

    def _request_range(self, start: int, end: int) -> bytes:
        request = urllib.request.Request(
            self._url,
            headers={
                "Range": f"bytes={start}-{end}",
                # Range 与 gzip 语义冲突，明确要求不做内容编码。
                "Accept-Encoding": "identity",
            },
        )
        with self._open_request(request) as response:
            status = _response_status(response)
            if status != 206:
                raise HTTPRangeError(
                    f"期望 206 Partial Content，实际 {status}；"
                    "服务器可能忽略了 Range 头，或该资源不支持按块读取"
                )
            r_start, r_end, total = _parse_content_range(
                _response_header(response, "Content-Range")
            )
            expected_end = min(end, total - 1)
            if r_start != start or r_end != expected_end:
                raise HTTPRangeError(
                    f"Content-Range 与请求错位：请求 bytes={start}-{end}，"
                    f"响应 bytes={r_start}-{r_end}/{total}"
                )
            expected = r_end - r_start + 1
            body = _read_bounded(response, expected + 1)
            if len(body) > expected:
                raise HTTPRangeError(
                    f"响应超出 Content-Range 声明：声明 {expected} 字节，"
                    f"实际至少 {len(body)} 字节"
                )
            if len(body) != expected:
                raise HTTPRangeError(
                    f"响应被截断：Content-Range 声明 {expected} 字节，实际 {len(body)} 字节"
                )
            if self._total is None:
                self._total = total
            elif self._total != total:
                raise HTTPRangeError(
                    f"资源总长度前后不一致：{self._total} → {total}"
                )
        return body

    def _gather(self, pos: int, count: int) -> bytes:
        """从 ``pos`` 起按块拼接至多 ``count`` 字节；EOF 处自然截断。"""
        total = self._ensure_total()
        out = bytearray()
        while len(out) < count and pos < total:
            index = pos // self._block_size
            block = self._get_block(index)
            if not block:
                break
            offset = pos - index * self._block_size
            if offset >= len(block):
                break
            take = min(count - len(out), len(block) - offset)
            out += block[offset : offset + take]
            pos += take
        return bytes(out)


# ----------------------------------------------------------------------
# 响应解析辅助
# ----------------------------------------------------------------------


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None:
        getcode = getattr(response, "getcode", None)
        if callable(getcode):
            status = getcode()
    if status is None:
        raise HTTPRangeError("无法从响应中取得 HTTP 状态码")
    return int(status)


def _response_header(response: Any, name: str) -> Optional[str]:
    getheader = getattr(response, "getheader", None)
    if callable(getheader):
        value = getheader(name)
        if value is not None:
            return value
    headers = getattr(response, "headers", None)
    if headers is not None:
        get = getattr(headers, "get", None)
        if callable(get):
            return get(name)
    return None


def _parse_content_range(value: Optional[str]) -> tuple[int, int, int]:
    if not value:
        raise HTTPRangeError("206 响应缺少 Content-Range 头")
    match = _CONTENT_RANGE_RE.match(value.strip())
    if match is None:
        raise HTTPRangeError(f"Content-Range 不合法: {value!r}")
    start, end, total = (int(group) for group in match.groups())
    if total <= 0 or start >= total or end < start or end >= total:
        raise HTTPRangeError(f"Content-Range 区间不合法: {value!r}")
    return start, end, total


def _read_bounded(response: Any, limit: int) -> bytes:
    """最多读 ``limit`` 字节；循环兜住单次 ``read`` 返回不足的流。"""
    chunks = []
    remaining = limit
    while remaining > 0:
        chunk = response.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
