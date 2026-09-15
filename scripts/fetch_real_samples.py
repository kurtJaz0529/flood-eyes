"""
慧眼识灾 · 真实卫星影像抓取
===========================

从 **AWS 公开 Sentinel-2 L2A COG 桶**（经 Element84 Earth Search STAC 检索）
按洪涝事件抓取"灾前 / 灾后"一对真实影像，直接落成项目可用的 4 波段 GeoTIFF。

为什么用这条链路？
    · 免注册、免密钥、免下载整个瓦片（HTTP Range 只读 10 km 窗口）
    · 数据是 Sentinel-2 L2A 大气校正反射率产品，10 m 分辨率
    · STAC 元数据自带 BOA 偏移量（2022 年后为 -1000），避免 NDWI 被系统性偏移毁掉

用法：
    python scripts/fetch_real_samples.py --list
    python scripts/fetch_real_samples.py --event dongting2024
    python scripts/fetch_real_samples.py --event all --size 1280
    python scripts/fetch_real_samples.py --event zhuozhou2023 --max-cloud 20

产物（data/real/）：
    {id}_pre.tif / {id}_post.tif   4 波段（B2 蓝 / B3 绿 / B4 红 / B8 近红外），反射率×10000
    {id}_pre_scl.png / _post_scl.png  场景分类图（水/云/云影/植被/其他）
    {id}_preview.jpg               灾前 / 灾后 真彩预览
    samples.json                   含景号、时间、云量、坐标等溯源信息
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

STAC_API = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"
USER_AGENT = "flood-eyes/0.1 (educational; contact: team)"

# --------------------------------------------------------------------------
# 双数据源
# --------------------------------------------------------------------------
# earthsearch：Element84 + AWS us-west-2 公开桶。免签名，但国内拉取 2MB 级
#   TIFF 块经常被掐断（Range 响应中途停流），表现为一到抓数就失败。
# pc：微软 Planetary Computer（Azure 西欧）。检索免密钥，资产读前用其公开
#   SAS 签名接口换一个带 token 的 URL；国内到 Azure 的链路稳定得多。
# 抓取时按 CATALOGS 顺序自动切换，哪个能读出数据用哪个。
PC_STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_SIGN_API = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
CATALOGS = ("pc", "earthsearch")
CATALOG_LABEL = {"pc": "Planetary Computer", "earthsearch": "Earth Search/AWS"}
_STAC_APIS = {"pc": PC_STAC_API, "earthsearch": STAC_API}
_BAND_KEYS = {
    "earthsearch": {"blue": "blue", "green": "green", "red": "red", "nir": "nir", "scl": "scl"},
    "pc": {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08", "scl": "SCL"},
}
_SIGN_CACHE: Dict[str, Tuple[str, float]] = {}
_SIGN_TTL_S = 3000.0  # PC 的 SAS token 约 1 小时有效，留余量提前重签
_SIGN_LOCK = threading.Lock()

# 允许发起请求的资产主机白名单。STAC 响应里的 href 会被交给 GDAL/rasterio
# 直接发起请求，若检索结果被污染或走了恶意代理，本机可能被诱导访问内网。
_ALLOWED_ASSET_HOSTS = (
    "sentinel-s2-l2a.s3.amazonaws.com",
    "sentinel-cogs.s3.us-west-2.amazonaws.com",
    "earth-search.aws.element84.com",
    "planetarycomputer.microsoft.com",
)

# 按后缀放行的域名。Planetary Computer 的 COG 资产放在 Azure Blob 上，
# 主机名带账号前缀且会变（sentinel2l2a01 / sentinel1euwestrtc01 / ai4edataeuwest …），
# 用精确匹配会把整个 PC 数据源拦死——必须按域后缀放行。
_ALLOWED_ASSET_HOST_SUFFIXES = (
    ".blob.core.windows.net",
)


def _catalog_of(item: Dict[str, Any]) -> str:
    return str(item.get("_catalog") or "earthsearch")


class AssetRejected(ValueError):
    """资产地址被协议/主机/IP 校验拒绝。

    刻意继承 ValueError（调用方既有的 except ValueError 仍能捕获），
    同时让重试逻辑能区分"确定性失败"与"网络抖动"。
    """


def _host_allowed(host: str) -> bool:
    """主机是否在允许的来源内（精确名单 + 受控域后缀）。

    单独抽成纯函数，便于离线验证白名单本身是否正确。
    """
    h = (host or "").lower()
    if not h:
        return False
    if h in _ALLOWED_ASSET_HOSTS:
        return True
    return any(h.endswith(suffix) for suffix in _ALLOWED_ASSET_HOST_SUFFIXES)


def _is_private_ip(ip_text: str) -> bool:
    """解析出的 IP 是否属于内网/环回/链路本地/保留段。"""
    import ipaddress

    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return True  # 解析不出的一律视为不可信
    return bool(
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_reserved or ip.is_multicast or ip.is_unspecified
    )


def _assert_public_https_url(raw_url: str) -> str:
    """发请求前的统一校验：协议 + 主机白名单 + 解析后 IP 不得指向内网。

    失败抛 `AssetRejected`（ValueError 子类）：这类失败是确定性的，
    调用方不应重试——重试只会白等几秒并刷日志。
    """
    parsed = urllib.parse.urlparse(str(raw_url))
    if parsed.scheme != "https":
        raise AssetRejected(f"拒绝非 https 地址：{str(raw_url)[:100]}")
    host = (parsed.hostname or "").lower()
    if not _host_allowed(host):
        raise AssetRejected(f"拒绝白名单之外的资产主机：{host or '(空主机)'}")
    import socket

    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise AssetRejected(f"资产主机无法解析：{host}") from exc
    for info in infos:
        addr = info[4][0]
        if _is_private_ip(addr):
            # 防 DNS rebinding / 被劫持的 DNS 把请求引向内网或云元数据地址
            raise AssetRejected(f"资产主机 {host} 解析到非公网地址 {addr}，已拒绝")
    return str(raw_url)


def write_json_atomic(path: str, payload: Any) -> str:
    """原子写入 JSON：先写同目录临时文件，再 os.replace 改名。

    直接覆写时若中途崩溃/断电，会留下截断的 JSON；下次加载解析失败只能在
    警告后按空清单处理，等于把之前抓好的样本记录整批丢掉。
    """
    from pathlib import Path

    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, str(path))
    return str(path)


def _sign_pc(href: str) -> str:
    """用 PC 公开签名接口换带 SAS token 的资产 URL（带 TTL 缓存）。

    两处加固：
    · href 来自 STAC 响应属外部输入，先过协议/主机/IP 校验再进请求；
    · 缓存带过期时间——token 失效后读取会持续 403，而"永不过期"的缓存
      会让程序再也无法自愈（原实现注释写了 1 小时复用，代码却永不清除）。
    """
    now = time.time()
    with _SIGN_LOCK:
        hit = _SIGN_CACHE.get(href)
        if hit and hit[1] > now:
            return hit[0]
    href = _assert_public_https_url(href)
    url = _assert_public_https_url(PC_SIGN_API + "?" + urllib.parse.urlencode({"href": href}))
    last: Optional[Exception] = None
    for i in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=20) as resp:
                signed = _assert_public_https_url(str(json.load(resp)["href"]))
            with _SIGN_LOCK:
                _SIGN_CACHE[href] = (signed, time.time() + _SIGN_TTL_S)
            return signed
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (i + 1))
    raise last or RuntimeError("PC 签名失败")


def _asset_href(item: Dict[str, Any], band: str) -> str:
    """按数据源取资产 URL；PC 资产先签名。"""
    cat = _catalog_of(item)
    href = str(item["assets"][_BAND_KEYS[cat][band]]["href"])
    return _sign_pc(href) if cat == "pc" else href

# SCL 类别 -> 颜色（用于可视化）
SCL_COLORS = {
    0: (0, 0, 0),        # 无数据
    1: (120, 120, 120),  # 饱和/缺陷
    2: (60, 60, 60),     # 暗区
    3: (150, 150, 150),  # 云影
    4: (60, 150, 60),    # 植被
    5: (200, 180, 120),  # 裸土
    6: (30, 90, 200),    # 水体
    7: (150, 200, 220),  # 未分类
    8: (235, 235, 235),  # 云（中概率）
    9: (255, 255, 255),  # 云（高概率）
    10: (210, 225, 245), # 卷云
    11: (240, 240, 240), # 雪
}
CLOUD_CLASSES = (3, 8, 9, 10)  # 云 + 云影

# --------------------------------------------------------------------------
# 洪涝事件注册表
# --------------------------------------------------------------------------
# pre/post 为 (起, 止, 目标日期)：脚本会在区间内挑"窗口云量最低 + 最接近目标日期"的景
EVENTS: Dict[str, Dict[str, Any]] = {
    "dongting2024": {
        "label": "2024·湖南洞庭湖团洲垸决口",
        "aoi": (112.66, 29.36),
        "pre": ("2024-06-01", "2024-06-30", "2024-06-20"),
        "post": ("2024-07-06", "2024-07-25", "2024-07-08"),
        "size": 1280,
        "note": "2024-07-05 17:48 团洲垸决口，垸内约 47 km² 被淹，7 月中旬完成排涝",
    },
    "zhuozhou2023": {
        "label": "2023·河北涿州暴雨洪涝",
        "aoi": (115.97, 39.49),
        "pre": ("2023-07-08", "2023-07-28", "2023-07-18"),
        "post": ("2023-08-01", "2023-08-22", "2023-08-05"),
        "size": 1280,
        "note": "2023-07-29~08-01 京津冀特大暴雨，涿州城区及周边大面积进水",
    },
    "poyang2020": {
        "label": "2020·江西鄱阳湖特大洪水",
        "aoi": (116.30, 29.15),
        "pre": ("2020-05-08", "2020-06-05", "2020-05-20"),
        "post": ("2020-07-10", "2020-07-30", "2020-07-13"),
        "size": 1280,
        "note": "2020-07 长江流域特大洪水，鄱阳湖水域面积较汛前扩大约 3 倍",
    },
    "rasuwa2026": {
        "label": "2026·尼泊尔拉苏瓦（Rasuwa）山洪 / 赛布鲁贝西",
        "aoi": (85.347, 28.163),
        "pre": ("2026-08-08", "2026-08-22", "2026-08-16"),
        "post": ("2026-08-26", "2026-09-02", "2026-08-28"),
        "size": 1280,
        "note": "2026-08-26 Rasuwa 山洪，赛布鲁贝西水电受损（Sentinel Asia EOR / GLIDE FF-2026-000162-NPL）",
    },
    "brazil2024": {
        "label": "2024·巴西南里奥格兰德州（阿雷格里港 / 瓜伊巴湖）",
        "aoi": (-51.31, -30.02),
        "pre": ("2024-04-01", "2024-04-20", "2024-04-03"),
        "post": ("2024-05-06", "2024-05-22", "2024-05-06"),
        "size": 1280,
        "note": "2024-05 巴西南部特大洪水，瓜伊巴湖漫溢，阿雷格里港及周边城镇大面积被淹",
    },
}

_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.TIF",
    "CPL_VSIL_CURL_USE_HEAD": "NO",
    "GDAL_HTTP_MULTIPLEX": "NO",
    "GDAL_HTTP_VERSION": "1",
    "GDAL_HTTP_TIMEOUT": "30",
    "GDAL_HTTP_CONNECTTIMEOUT": "10",
    "GDAL_HTTP_MAX_RETRY": "2",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "AWS_NO_SIGN_REQUEST": "YES",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "67108864",
}

# 界面调用时缩短等待：到点立刻放弃，改走本地缓存
_FAST = False
_DEADLINE: Optional[float] = None


def _check_deadline() -> None:
    if _DEADLINE is not None and time.time() > _DEADLINE:
        raise TimeoutError("在线下载超时")


def _gdal_env() -> Dict[str, str]:
    env = dict(_GDAL_ENV)
    if _DEADLINE is not None:
        left = max(8.0, _DEADLINE - time.time())
        env["GDAL_HTTP_TIMEOUT"] = str(int(min(30.0, left)))
    return env


# --------------------------------------------------------------------------
# STAC
# --------------------------------------------------------------------------


def stac_search(
    bbox: List[float],
    datetime_range: str,
    limit: int = 200,
    cloud_lt: Optional[float] = None,
    catalog: str = "earthsearch",
) -> List[Dict[str, Any]]:
    """检索 Sentinel-2 L2A 条目。bbox = [minx, miny, maxx, maxy]（经纬度）。

    catalog = "earthsearch"（Element84/AWS）或 "pc"（微软 Planetary Computer）。
    返回的每个条目都打上 `_catalog` 标记，后续取资产时按数据源分流。
    """
    body: Dict[str, Any] = {
        "collections": [COLLECTION],
        "bbox": bbox,
        "datetime": datetime_range,
        "limit": limit,
    }
    if catalog == "earthsearch":
        body["sortby"] = [{"field": "properties.datetime", "direction": "asc"}]
    if cloud_lt is not None:
        body["query"] = {"eo:cloud_cover": {"lt": cloud_lt}}
    req = urllib.request.Request(
        _STAC_APIS.get(catalog, STAC_API),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    attempts = 1 if _FAST else 3
    timeout = 12 if _FAST else 60
    for attempt in range(attempts):
        _check_deadline()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                feats = json.load(resp).get("features", [])
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            print(f"  [retry] STAC 检索失败 {type(exc).__name__}，{2 * (attempt + 1)} 秒后重试")
            time.sleep(2 * (attempt + 1))
            continue
        for f in feats:
            f["_catalog"] = catalog
        return feats
    return []


def point_in_bbox(bbox: List[float], lon: float, lat: float) -> bool:
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def _with_retry(fn, attempts: int = 3, base_delay: float = 2.0, label: str = ""):
    """网络抖动重试。

    AWS 公开 COG 走 `/vsicurl/` 的 HTTP Range 读取会间歇性失败
    （连接被重置 / 返回体截断），偶发失败不该直接判一景报废。
    """
    last: Optional[Exception] = None
    for i in range(attempts):
        try:
            return fn()
        except AssetRejected:
            # 协议/主机/IP 校验拒绝是确定性失败，重试不会改变结果。
            # 不在这里放行的话，每个被拒的候选景都要白等 2+4 秒并刷日志。
            raise
        except TimeoutError:
            # deadline 到点是调用方的硬约束，不能被当成"网络抖动"再等一轮
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            if i < attempts - 1:
                wait = base_delay * (i + 1)
                print(f"    [retry] {label} 读取失败({type(exc).__name__})，{wait:.0f} 秒后重试")
                time.sleep(wait)
    if last is None:
        # 原实现用 assert 做不可达假设，python -O 下被剥离后会抛 TypeError 掩盖真因
        raise RuntimeError(f"{label or '读取'} 未执行成功且未记录异常")
    raise last


# --------------------------------------------------------------------------
# COG 窗口读取
# --------------------------------------------------------------------------


def _open_vsicurl(href: str):
    import rasterio

    url = href if href.startswith("/vsicurl/") else "/vsicurl/" + href
    try:
        return rasterio.open(url)
    except Exception:
        return rasterio.open(href.replace("/vsicurl/", "", 1) if href.startswith("/vsicurl/") else href)


def _utm_bounds(src_crs: Any, lon: float, lat: float, half_km: float) -> Tuple[float, float, float, float]:
    from rasterio.warp import transform as warp_transform

    xs, ys = warp_transform("EPSG:4326", src_crs, [lon], [lat])
    x, y = float(xs[0]), float(ys[0])
    half = half_km * 1000.0
    return x - half, y - half, x + half, y + half


def read_window(href: str, crs: Any, bounds: Tuple[float, float, float, float],
                out_size: Optional[int] = None, dtype: str = "float32") -> Tuple[np.ndarray, Any]:
    """按地理范围读取 COG 窗口。大窗口拆成 128px 小块，避免一次 Range 请求过大失败。

    国内访问 us-west-2 的 S3 链路抖动明显：同一连接一旦读坏（Range 响应被截断），
    后续读会立即失败。所以除了单块重试，还加了"整轮重开"——失败时关掉句柄、
    重建 Env 再从未完成的分块继续，已完成的分块跨轮保留。
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window, from_bounds

    _check_deadline()
    out: Optional[np.ndarray] = None
    done: set = set()
    last: Optional[Exception] = None
    for round_i in range(3):
        try:
            with rasterio.Env(**_gdal_env()):
                with _open_vsicurl(href) as src:
                    win = from_bounds(*bounds, transform=src.transform)
                    full = Window(0, 0, src.width, src.height)
                    win = win.intersection(full)
                    if win.width <= 1 or win.height <= 1:
                        raise ValueError("窗口落在影像有效范围外")
                    src_h, src_w = float(win.height), float(win.width)
                    if out_size:
                        dst_h = dst_w = int(out_size)
                    else:
                        dst_h, dst_w = max(1, int(round(src_h))), max(1, int(round(src_w)))
                    if out is None:
                        out = np.zeros((dst_h, dst_w), dtype=np.dtype(dtype))
                    tile = 128 if dst_h * dst_w > 128 * 128 else max(dst_h, dst_w)
                    n_y = (dst_h + tile - 1) // tile
                    n_x = (dst_w + tile - 1) // tile
                    n_tiles = n_y * n_x
                    for iy in range(n_y):
                        for ix in range(n_x):
                            if (iy, ix) in done:
                                continue
                            _check_deadline()
                            y0, x0 = iy * tile, ix * tile
                            th, tw = min(tile, dst_h - y0), min(tile, dst_w - x0)
                            sy0 = win.row_off + y0 * src_h / dst_h
                            sx0 = win.col_off + x0 * src_w / dst_w
                            sw = Window(sx0, sy0, tw * src_w / dst_w, th * src_h / dst_h)
                            block = None
                            tile_exc: Optional[Exception] = None
                            for attempt in range(5):
                                try:
                                    block = src.read(
                                        1,
                                        window=sw,
                                        out_shape=(th, tw),
                                        resampling=Resampling.bilinear,
                                        boundless=True,
                                        fill_value=0,
                                    )
                                    break
                                except Exception as exc:
                                    tile_exc = exc
                                    time.sleep(1.5 * (attempt + 1))
                            if block is None:
                                raise tile_exc or RuntimeError("分块读取失败")
                            out[y0 : y0 + th, x0 : x0 + tw] = block.astype(dtype, copy=False)
                            done.add((iy, ix))
                            if len(done) == 1 or len(done) == n_tiles or len(done) % 4 == 0:
                                print(f"      分块 {len(done)}/{n_tiles}")
                    return out, src.window_transform(win)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if isinstance(exc, TimeoutError):
                raise
            wait = 3.0 * (round_i + 1)
            print(f"    [retry] 整窗读取中断（{type(exc).__name__}），{wait:.0f} 秒后重开连接"
                  f"（已完成 {len(done)} 块保留）")
            time.sleep(wait)
    assert last is not None
    raise last


def find_valid_center(
    item: Dict[str, Any],
    lon: float,
    lat: float,
    max_shift_km: float = 4.0,
) -> Optional[Tuple[float, float, float]]:
    """把窗口中心从 AOI 挪到最近的"有效像元"。

    Sentinel-2 瓦片在 UTM 下是旋转的，瓦片包围盒的角上是大片 nodata；
    如果 AOI 恰好落在角上，直接按经纬度取窗口会读出一片 0。
    这里用 SCL 的整幅缩略图找最近的有效像元，返回 (x_utm, y_utm, 偏移公里)。
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform as warp_transform

    with rasterio.Env(**_gdal_env()):
        with _open_vsicurl(_asset_href(item, "scl")) as src:
            xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
            x, y = float(xs[0]), float(ys[0])
            col, row = ~src.transform * (x, y)  # 全分辨率像素坐标
            ov_size = 256
            ov = src.read(1, out_shape=(ov_size, ov_size), resampling=Resampling.nearest)
            crs, transform, width, height = src.crs, src.transform, src.width, src.height

    valid = ov > 0
    if not valid.any():
        return None
    # AOI 在缩略图上的位置
    oc = int(round(col * ov_size / width))
    orow = int(round(row * ov_size / height))
    if 0 <= oc < ov_size and 0 <= orow < ov_size and valid[orow, oc]:
        return x, y, 0.0
    # 找最近的有效缩略图像元
    vy, vx = np.nonzero(valid)
    dist2 = (vy - orow) ** 2 + (vx - oc) ** 2
    i = int(np.argmin(dist2))
    # 缩略图像元 -> 全分辨率 -> UTM 坐标
    px = (vx[i] + 0.5) * width / ov_size
    py = (vy[i] + 0.5) * height / ov_size
    nx, ny = transform * (px, py)
    shift_km = float(np.hypot(nx - x, ny - y) / 1000.0)
    if shift_km > max_shift_km:
        return None
    return float(nx), float(ny), shift_km


def scene_window_quality(item: Dict[str, Any], x_utm: float, y_utm: float, half_km: float = 5.0) -> Dict[str, float]:
    """用 SCL 评估窗口的云量/水体/有效率（低分辨率快读）。"""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    with rasterio.Env(**_gdal_env()):
        with _open_vsicurl(_asset_href(item, "scl")) as src:
            win = from_bounds(x_utm - half_km * 1000, y_utm - half_km * 1000,
                              x_utm + half_km * 1000, y_utm + half_km * 1000,
                              transform=src.transform)
            scl = src.read(1, window=win, out_shape=(128, 128), resampling=Resampling.nearest)
    total = max(scl.size, 1)
    return {
        "cloud_pct": 100.0 * float(np.isin(scl, CLOUD_CLASSES).sum()) / total,
        "water_pct": 100.0 * float((scl == 6).sum()) / total,
        "valid_pct": 100.0 * float((scl > 0).sum()) / total,
    }


# --------------------------------------------------------------------------
# 事件抓取
# --------------------------------------------------------------------------


def _pick_scene(
    candidates: List[Dict[str, Any]],
    lon: float,
    lat: float,
    target_date: str,
    max_cloud: float,
    half_km: float,
    allow_cloudy: bool = False,
) -> Optional[Tuple[Dict[str, Any], Dict[str, float], Tuple[float, float, float]]]:
    """在候选景中挑"有效数据 + 窗口云量低 + 时间接近目标"的一景。"""
    scored: List[Tuple[float, Dict[str, Any], Dict[str, float], Tuple[float, float, float]]] = []
    n_err = 0
    for item in candidates:
        if not point_in_bbox(item["bbox"], lon, lat):
            continue
        date = item["properties"]["datetime"][:10]
        # ① 定位窗口中心的有效像元
        try:
            center = _with_retry(lambda: find_valid_center(item, lon, lat), label=item["id"])
        except Exception as exc:
            n_err += 1
            print(f"    [skip] {item['id']}: {type(exc).__name__} {str(exc)[:100]}")
            continue
        if center is None:
            print(f"    {date}  {item['id']:34s}  ✗ AOI 附近无有效数据")
            continue
        x, y, shift = center
        # ② 评估窗口云量 / 水体 / 有效率
        try:
            q = _with_retry(lambda: scene_window_quality(item, x, y, half_km=half_km), label=item["id"])
        except Exception as exc:
            n_err += 1
            print(f"    [skip] {item['id']}: {type(exc).__name__} {str(exc)[:100]}")
            continue
        if q["valid_pct"] < 50:
            print(f"    {date}  {item['id']:34s}  ✗ 有效率仅 {q['valid_pct']:.0f}%")
            continue
        dt_days = abs((np.datetime64(date) - np.datetime64(target_date)) / np.timedelta64(1, "D"))
        score = q["cloud_pct"] + 1.5 * dt_days + 2.0 * shift
        scored.append((score, item, q, center))
        print(f"    {date}  {item['id']:34s} 窗口云量 {q['cloud_pct']:5.1f}%  "
              f"水体 {q['water_pct']:5.1f}%  有效 {q['valid_pct']:3.0f}%  "
              f"挪窗 {shift:4.1f}km  评分 {score:6.1f}")
    if not scored:
        if n_err:
            print(f"    ✗ {n_err} 景因网络读取失败被跳过（已自动重试仍未成功，可稍后再试）")
        return None
    scored.sort(key=lambda x: x[0])
    best = scored[0]
    if best[2]["cloud_pct"] > max_cloud:
        msg = (f"最优景窗口云量 {best[2]['cloud_pct']:.1f}% 超过阈值 {max_cloud}%，"
               f"说明该时段光学影像被云遮挡")
        if not allow_cloudy:
            print(f"    ✗ {msg}。可换事件/日期，或加 --allow-cloudy 强制使用")
            return None
        print(f"    ⚠ {msg}（--allow-cloudy 已强制使用）")
    return best[1], best[2], best[3]


def _point_in_ring(lon: float, lat: float, ring: List[Any]) -> bool:
    """射线法判断点是否在经纬度多边形内（一景范围内平面近似足够）。"""
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = float(ring[i][0]), float(ring[i][1])
        x2, y2 = float(ring[(i + 1) % n][0]), float(ring[(i + 1) % n][1])
        if (y1 > lat) != (y2 > lat):
            xin = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            if lon < xin:
                inside = not inside
    return inside


def _covers_aoi(item: Dict[str, Any], lon: float, lat: float) -> bool:
    """用 STAC 数据覆盖多边形精确判断该景是否真的覆盖 AOI。

    item["bbox"] 只是覆盖多边形的外接矩形：Sentinel-2 条带是旋转的，
    bbox 角上常有大片 nodata，只查 bbox 会误选"看着覆盖、实际没数据"的景。
    """
    geom = item.get("geometry") or {}
    gtype = geom.get("type")
    coords = geom.get("coordinates") or []
    try:
        if gtype == "Polygon" and coords:
            return _point_in_ring(lon, lat, coords[0])
        if gtype == "MultiPolygon":
            return any(bool(poly) and _point_in_ring(lon, lat, poly[0]) for poly in coords)
    except Exception:
        pass
    return True  # 没有几何信息时不误杀


def _pick_scene_stac(
    candidates: List[Dict[str, Any]],
    lon: float,
    lat: float,
    target_date: str,
    max_cloud: float,
    allow_cloudy: bool = False,
    half_km: float = 2.56,
    verify_k: int = 8,
) -> Optional[Tuple[
    Tuple[Dict[str, Any], Dict[str, float], Tuple[float, float, float]],
    List[Tuple[Dict[str, Any], Dict[str, float], Tuple[float, float, float]]],
]]:
    """按 STAC 元数据初排，再逐景验证"AOI 处有有效像元"，返回 (首选, 备选列表)。

    快速路径（fast=True）不能用 bbox 粗筛就直接下载：条带边缘的景在 AOI 处
    全是 nodata，会抓回一整块全零影像。这里对初排前 verify_k 个候选各读一次
    SCL 全图缩略图（1 个小请求/景，find_valid_center），确认覆盖并把窗口中心
    挪到最近的有效像元；验证不通过的景直接淘汰，不会进入抓取环节。
    """
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for item in candidates:
        # STAC 条目允许缺 bbox / datetime / 云量，缺字段时应淘汰该景，
        # 而不是让 KeyError/TypeError 中断整轮候选遍历。
        bbox_item = item.get("bbox")
        if not bbox_item:
            continue
        if not point_in_bbox(bbox_item, lon, lat):
            continue
        if not _covers_aoi(item, lon, lat):
            continue
        props = item.get("properties") or {}
        date = str(props.get("datetime") or "")[:10]
        if len(date) < 10:
            continue
        try:
            np.datetime64(date)
        except (ValueError, TypeError):
            print(f"    [skip] {item.get('id', '?')}: 日期字段非法（{date!r}）")
            continue
        cloud = float(props.get("eo:cloud_cover") or 100.0)
        dt_days = abs((np.datetime64(date) - np.datetime64(target_date)) / np.timedelta64(1, "D"))
        scored.append((cloud + 1.5 * float(dt_days), item))
    if not scored:
        print("    ✗ 没有候选景的数据覆盖多边形包含 AOI")
        return None
    scored.sort(key=lambda x: x[0])

    verified: List[Tuple[float, Dict[str, Any], Dict[str, float], Tuple[float, float, float]]] = []
    for score, item in scored[: max(1, int(verify_k))]:
        date = str((item.get("properties") or {}).get("datetime", ""))[:10]
        try:
            center = _with_retry(lambda: find_valid_center(item, lon, lat), label=item["id"])
        except Exception as exc:
            print(f"    [skip] {item['id']}: {type(exc).__name__} {str(exc)[:80]}")
            continue
        if center is None:
            print(f"    {date}  {item['id']:34s}  ✗ AOI 附近无有效数据")
            continue
        cloud = float((item.get("properties") or {}).get("eo:cloud_cover", 100.0))
        quality = {"cloud_pct": cloud, "water_pct": 0.0, "valid_pct": 100.0}
        try:
            quality = scene_window_quality(item, center[0], center[1], half_km=half_km)
        except Exception as exc:
            # 质检失败不能沿用 valid_pct=100 的默认值：那等于把"没测出来"当成
            # "满分景"，随后的云量判断建立在伪造数据上，可能把重云景当成首选。
            print(f"    [skip] {item['id']}: 窗口质检失败 {type(exc).__name__}，淘汰该景")
            continue
        verified.append((score, item, quality, center))
        print(f"    {date}  {item['id']:34s} 景云量 {cloud:5.1f}%  窗口云量 {quality['cloud_pct']:5.1f}%  "
              f"挪窗 {center[2]:4.1f}km  评分 {score:6.1f}")
    if not verified:
        print("    ✗ 排名靠前的候选景在 AOI 处都无有效数据")
        return None
    verified.sort(key=lambda x: x[0])
    best = verified[0]
    if best[2]["cloud_pct"] > max_cloud and not allow_cloudy:
        cloudy = [v for v in verified if v[2]["cloud_pct"] <= 80.0]
        if not cloudy:
            print(f"    ✗ 最优景云量 {best[2]['cloud_pct']:.1f}% 过高")
            return None
        best = cloudy[0]
    picked = (best[1], best[2], best[3])
    ranked = [(v[1], v[2], v[3]) for v in verified]
    return picked, ranked


def _reflectance_scale(item: Dict[str, Any], band: str) -> Tuple[float, float]:
    """从 STAC 元数据读 scale/offset（2022 年后 offset = -0.1）。

    PC 条目的资产通常不带 raster:bands，且像元是未平移的原始 DN：
    baseline ≥ 04.00 时按 ESA 规则补 offset=-0.1，否则为 0。
    """
    cat = _catalog_of(item)
    key = _BAND_KEYS[cat].get(band, band)
    rb = (item["assets"].get(key, {}).get("raster:bands") or [{}])[0]
    scale = float(rb.get("scale", 0.0001) or 0.0001)
    offset = float(rb.get("offset", 0.0) or 0.0)
    if cat == "pc" and offset == 0.0:
        try:
            baseline = float(str((item.get("properties") or {}).get("s2:processing_baseline", "0")) or 0)
        except ValueError:
            baseline = 0.0
        if baseline >= 4.0:
            offset = -0.1
    return scale, offset


_BAND_TO_JP2 = {"blue": ("B02", "R10m"), "green": ("B03", "R10m"),
                "red": ("B04", "R10m"), "nir": ("B08", "R10m")}


def jp2_href(item: Dict[str, Any], band: str) -> Optional[str]:
    """由 STAC 条目 id 拼出原始 JP2（ESA 官方产品）地址，用于偏移量权威判定。"""
    try:
        parts = item["id"].split("_")  # S2A_50RMT_20200519_1_L2A
        mgrs, date, seq = parts[1], parts[2], parts[3]
        zone, latband, square = mgrs[:2], mgrs[2], mgrs[3:5]
        band_id, res = _BAND_TO_JP2[band]
        return (f"https://sentinel-s2-l2a.s3.amazonaws.com/tiles/{zone}/{latband}/{square}/"
                f"{date[:4]}/{int(date[4:6])}/{int(date[6:8])}/{seq}/{res}/{band_id}.jp2")
    except Exception:
        return None


def detect_offset_applied(
    item: Dict[str, Any], crs: Any, bounds: Tuple[float, float, float, float]
) -> Tuple[bool, str]:
    """判定 COG 是否已经应用了 BOA 偏移（baseline>=04.00 的 -1000）。

    背景：Element84 的 `earthsearch:boa_offset_applied` 标记存在已知错误
    （issue #66/#71），官方建议与原始 JP2 比对。这里就用 JP2 做权威判定：
        同一窗口 |中位DN_COG - 中位DN_JP2| ≈ 1000  ->  COG 已扣偏移
                                       ≈ 0      ->  COG 未扣偏移
    读 JP2 失败时退回 STAC 标记。
    """
    band = "red"
    _scale, offset = _reflectance_scale(item, band)
    if offset == 0.0:  # baseline < 04.00，本来就没有偏移
        return False, "无偏移(baseline<04.00)"
    href = None  # JP2 源在国内经常不可达，一律用 STAC 标记
    if href:
        try:
            jp2, _ = read_window(href, crs, bounds, out_size=128)
            cog, _ = read_window(item["assets"][band]["href"], crs, bounds, out_size=128)
            diff = float(np.median(cog) - np.median(jp2))
            print(f"    [offset] COG中位DN={np.median(cog):.0f}  JP2中位DN={np.median(jp2):.0f}  Δ={diff:+.0f}")
            return abs(diff) > 600, f"JP2比对(ΔDN={diff:.0f})"
        except Exception as exc:
            print(f"    [warn] JP2 比对失败({type(exc).__name__})，退回 STAC 标记")
    flag = bool(item["properties"].get("earthsearch:boa_offset_applied", False))
    return flag, f"STAC标记(boa_offset_applied={flag})"


def reflectance_from_dn(dn: np.ndarray, scale: float, offset: float, offset_applied: bool) -> np.ndarray:
    """DN -> 反射率。COG 已扣过偏移时把 offset 置 0，避免二次扣除。"""
    eff_offset = 0.0 if offset_applied else offset
    return dn.astype(np.float32) * scale + eff_offset


def fetch_scene(
    item: Dict[str, Any],
    x_utm: float,
    y_utm: float,
    size: int,
    out_tif: str,
) -> Dict[str, Any]:
    """抓取一景的 4 波段窗口并落盘为 GeoTIFF（窗口以 x_utm/y_utm 为中心）。"""
    return _fetch_scene_at(item, x_utm, y_utm, int(size), out_tif)


def _fetch_scene_at(
    item: Dict[str, Any],
    x_utm: float,
    y_utm: float,
    size: int,
    out_tif: str,
) -> Dict[str, Any]:
    import rasterio

    cat = _catalog_of(item)
    half_km = size * 10.0 / 2000.0  # 10 m 像元 -> 公里
    bounds = (x_utm - half_km * 1000, y_utm - half_km * 1000,
              x_utm + half_km * 1000, y_utm + half_km * 1000)
    with rasterio.Env(**_gdal_env()):
        with _open_vsicurl(_asset_href(item, "blue")) as src:
            crs = src.crs

        if cat == "pc":
            # PC 像元是未平移的原始 DN，偏移规则由 baseline 决定，无需判定
            offset_applied, offset_method = False, "PC 原始 DN（按 s2:processing_baseline 扣偏移）"
        else:
            offset_applied, offset_method = detect_offset_applied(item, crs, bounds)
        print(f"    [offset] 判定 COG 已扣偏移={offset_applied}  依据：{offset_method}  窗口={size}")

        bands: Dict[str, np.ndarray] = {}
        valid_frac = 0.0  # 在 clip 之前用原始 DN 统计，nodata=0
        for band in ("blue", "green", "red", "nir"):
            scale, offset = _reflectance_scale(item, band)
            raw, wtransform = read_window(_asset_href(item, band), crs, bounds, out_size=size)
            if band == "blue":
                valid_frac = float(np.count_nonzero(raw) / max(raw.size, 1))
            refl = reflectance_from_dn(raw, scale, offset, offset_applied)
            # 落盘为"反射率 ×10000"的干净产品，供 preprocess.to_reflectance 直接除 10000。
            # 必须保留 nodata=0：把 0 一起 clip 到下限 1，会让无效像元变成
            # 1e-4 反射率的"有效暗像元"，与 profile 里声明的 nodata=0 自相矛盾，
            # 条带边缘的大片 nodata 会以极暗地物身份混进水体判定。
            scaled = np.clip(refl * 10000.0, 1, 10000)
            scaled[raw == 0] = 0.0
            bands[band] = scaled.astype(np.uint16)
            transform = wtransform
            neg = 100.0 * float((refl < 0).mean())
            print(f"    {band:6s} scale={scale:g} offset={offset:+.3f}  "
                  f"中位 DN={np.median(raw):7.1f} -> 反射率 {np.median(refl):.4f}  负值占比 {neg:4.1f}%")

        if valid_frac < 0.5:
            raise ValueError(
                f"窗口有效像元仅 {valid_frac*100:.0f}%，该景实际未覆盖目标区域（nodata），不写盘"
            )

        scl, _ = read_window(_asset_href(item, "scl"), crs, bounds, out_size=size,
                             dtype="uint8")
        arr = np.stack([bands["blue"], bands["green"], bands["red"], bands["nir"]], axis=0)

    profile = {
        "driver": "GTiff", "height": size, "width": size, "count": 4, "dtype": "uint16",
        "crs": crs, "transform": transform, "compress": "deflate", "predictor": 2,
        "nodata": 0, "tiled": True, "blockxsize": 256, "blockysize": 256,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_tif)), exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as ds:
        ds.write(arr)
        for i, name in enumerate(("B2(blue)", "B3(green)", "B4(red)", "B8(nir)"), start=1):
            ds.set_band_description(i, name)

    total = max(scl.size, 1)
    return {
        "scl": scl,
        "cloud_pct": 100.0 * float(np.isin(scl, CLOUD_CLASSES).sum()) / total,
        "water_pct": 100.0 * float((scl == 6).sum()) / total,
        "valid_pct": 100.0 * valid_frac,
        "offset_applied": offset_applied,
        "offset_method": offset_method,
        "arr": arr,
    }


def save_scl_png(scl: np.ndarray, path: str) -> str:
    from PIL import Image

    rgb = np.zeros((*scl.shape, 3), dtype=np.uint8)
    for k, color in SCL_COLORS.items():
        rgb[scl == k] = color
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(rgb).save(path)
    return path


def save_preview(pre_arr: np.ndarray, post_arr: np.ndarray, path: str, max_width: int = 1100) -> str:
    """灾前/灾后真彩预览（B4,B3,B2 -> RGB，联合百分位拉伸）。"""
    from PIL import Image

    def to_rgb(a: np.ndarray) -> np.ndarray:
        return np.moveaxis(a[[2, 1, 0]].astype(np.float32), 0, -1)

    pre_rgb, post_rgb = to_rgb(pre_arr), to_rgb(post_arr)
    lo, hi = np.percentile(np.concatenate([pre_rgb.ravel(), post_rgb.ravel()]), [2, 98])
    norm = lambda x: (np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)  # noqa: E731
    gap = np.full((pre_rgb.shape[0], 8, 3), 255, dtype=np.uint8)
    canvas = np.concatenate([norm(pre_rgb), gap, norm(post_rgb)], axis=1)
    im = Image.fromarray(canvas)
    if im.width > max_width:
        im = im.resize((max_width, int(im.height * max_width / im.width)), Image.LANCZOS)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    im.save(path, quality=88, optimize=True)
    return path


def _cached_pair(out_dir: str, key: str, size: int) -> Optional[Tuple[str, str]]:
    """本地已抓过同一 AOI + 同窗口尺寸，直接复用，省一次联网。"""
    pre = os.path.join(out_dir, f"{key}_pre.tif")
    post = os.path.join(out_dir, f"{key}_post.tif")
    if not (os.path.isfile(pre) and os.path.isfile(post)):
        return None
    try:
        import rasterio

        with rasterio.open(pre) as ds:
            if int(ds.width) != int(size) or int(ds.height) != int(size):
                return None
    except Exception:
        return None
    return pre, post


def fetch_event(
    key: str,
    cfg: Dict[str, Any],
    out_dir: str,
    size: Optional[int] = None,
    max_cloud: float = 25.0,
    allow_cloudy: bool = False,
    fast: bool = False,
    budget_s: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """抓取一个事件的一对灾前/灾后影像。"""
    global _FAST, _DEADLINE
    _FAST = bool(fast)
    _DEADLINE = (time.time() + float(budget_s)) if budget_s else None
    try:
        return _fetch_event_body(key, cfg, out_dir, size, max_cloud, allow_cloudy)
    except TimeoutError as exc:
        print(f"  ✗ {exc}")
        return None
    finally:
        _FAST = False
        _DEADLINE = None


def _fetch_event_body(
    key: str,
    cfg: Dict[str, Any],
    out_dir: str,
    size: Optional[int],
    max_cloud: float,
    allow_cloudy: bool,
) -> Optional[Dict[str, Any]]:
    lon, lat = cfg["aoi"]
    size = int(size or cfg.get("size", 1280))
    half_km = size * 10.0 / 2000.0  # 质检窗口与最终抓取窗口保持同尺寸
    pad = 0.25  # 检索框略大于 AOI，确保覆盖
    bbox = [lon - pad, lat - pad, lon + pad, lat + pad]

    cached = _cached_pair(out_dir, key, size)
    if cached is not None:
        print("=" * 78)
        print(f"事件：{cfg['label']}   AOI=({lon}, {lat})   窗口 {size}×{size}")
        print(f"本地已缓存该 AOI 影像，跳过下载："
              f"{os.path.basename(cached[0])} / {os.path.basename(cached[1])}")
        return {
            "id": key,
            "label": cfg["label"],
            "description": cfg["note"],
            "aoi": [lon, lat],
            "size": [size, size],
            "pixel_size_m": 10.0,
            "pre": f"{key}_pre.tif",
            "post": f"{key}_post.tif",
            "preview": f"{key}_preview.jpg",
            "provenance": {"source": "本地缓存", "note": "复用本机已抓取的同一 AOI 影像"},
        }

    print("=" * 78)
    print(f"事件：{cfg['label']}   AOI=({lon}, {lat})   窗口 {size}×{size}（{size*10/1000:.1f} km）")
    print(f"说明：{cfg['note']}")
    print("-" * 78)

    entry: Dict[str, Any] = {
        "id": key,
        "label": cfg["label"],
        "description": cfg["note"],
        "aoi": [lon, lat],
        "size": [size, size],
        "pixel_size_m": 10.0,
        "pre": f"{key}_pre.tif",
        "post": f"{key}_post.tif",
        "preview": f"{key}_preview.jpg",
        "pre_scl": f"{key}_pre_scl.png",
        "post_scl": f"{key}_post_scl.png",
        "provenance": {},
    }

    for tag in ("pre", "post"):
        got = _fetch_tag(
            tag, cfg, lon, lat, bbox, size, half_km,
            max_cloud, allow_cloudy, out_dir, entry,
        )
        if got is None:
            print(f"  ✗ {tag} 两个数据源都没能拿到可用影像")
            return None
        item, quality, center, data, cat = got
        save_scl_png(data["scl"], os.path.join(out_dir, entry[f"{tag}_scl"]))
        entry[f"{tag}_arr"] = data["arr"]
        entry["provenance"][f"{tag}_scene"] = item["id"]
        entry["provenance"][f"{tag}_datetime"] = item["properties"]["datetime"]
        entry["provenance"][f"{tag}_scene_cloud_pct"] = round(float(item["properties"]["eo:cloud_cover"]), 1)
        entry["provenance"][f"{tag}_window_cloud_pct"] = round(quality["cloud_pct"], 1)
        entry["provenance"][f"{tag}_window_water_pct"] = round(quality["water_pct"], 1)
        entry["provenance"][f"{tag}_window_shift_km"] = round(center[2], 2)
        entry["provenance"][f"{tag}_epsg"] = int(item["properties"]["proj:epsg"])
        entry["provenance"][f"{tag}_boa_offset_applied"] = bool(data.get("offset_applied", False))
        entry["provenance"][f"{tag}_offset_basis"] = data.get("offset_method", "-")
        entry["provenance"][f"{tag}_catalog"] = CATALOG_LABEL.get(cat, cat)

    save_preview(entry.pop("pre_arr"), entry.pop("post_arr"), os.path.join(out_dir, entry["preview"]))

    entry["provenance"].update({
        "source": "Sentinel-2 L2A COG · 微软 Planetary Computer / Element84 Earth Search（自动切换）",
        "license": "Copernicus Sentinel Data Terms and Conditions (free and open)",
        "fetch_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "反射率已按 STAC 元数据应用 scale/offset，落盘为 反射率×10000 的 uint16",
    })
    return entry


def _fetch_tag(
    tag: str,
    cfg: Dict[str, Any],
    lon: float,
    lat: float,
    bbox: List[float],
    size: int,
    half_km: float,
    max_cloud: float,
    allow_cloudy: bool,
    out_dir: str,
    entry: Dict[str, Any],
) -> Optional[Tuple[Dict[str, Any], Dict[str, float], Tuple[float, float, float], Dict[str, Any], str]]:
    """抓一个时相：检索 → 选景 → 抓取；一个数据源不行就整链路换下一个。"""
    start, end, target = cfg[tag]
    last_exc: Optional[Exception] = None
    for cat in CATALOGS:
        _check_deadline()
        print(f"[{tag}] {CATALOG_LABEL.get(cat, cat)} 检索 {start} ~ {end}（目标 {target}）")
        try:
            items = stac_search(bbox, f"{start}T00:00:00Z/{end}T23:59:59Z", catalog=cat)
        except TimeoutError:
            # 预算到点必须整体上抛，不能当成"这个源不行"换下一个继续联网
            raise
        except Exception as exc:
            last_exc = exc
            print(f"  ✗ 检索失败 {type(exc).__name__}: {exc}")
            continue
        if _FAST:
            print(f"  候选 {len(items)} 景，按目录云量初排并逐景验证 AOI 覆盖：")
            got = _pick_scene_stac(items, lon, lat, target, max_cloud, allow_cloudy, half_km=half_km)
            ranked: List[Tuple[Any, Any, Any]] = list(got[1]) if got else []
        else:
            print(f"  候选 {len(items)} 景，逐个评估 AOI 窗口云量：")
            picked = _pick_scene(items, lon, lat, target, max_cloud, half_km, allow_cloudy)
            if picked is None:
                print("  本轮未选出可用景，重新检索再试一次…")
                items = stac_search(bbox, f"{start}T00:00:00Z/{end}T23:59:59Z", catalog=cat)
                print(f"  候选 {len(items)} 景，重新评估：")
                picked = _pick_scene(items, lon, lat, target, max_cloud, half_km, allow_cloudy)
            ranked = [picked] if picked else []
        if not ranked:
            print("  ✗ 没有找到覆盖 AOI 的可用景，换数据源")
            continue
        item0, quality0, _ = ranked[0]
        print(f"  ✓ 首选 {item0['id']}  {item0['properties']['datetime'][:16]}  "
              f"云量 {quality0['cloud_pct']:.1f}%")
        for item, quality, center in ranked:
            print(f"[{tag}] 抓取 {item['id']}  4 波段 + SCL（分块）...")
            t0 = time.time()
            try:
                data = fetch_scene(item, center[0], center[1], size, os.path.join(out_dir, entry[tag]))
            except TimeoutError:
                raise
            except Exception as exc:
                last_exc = exc
                print(f"  ✗ {item['id']} 失败 {type(exc).__name__}: {str(exc)[:120]}，换下一景")
                continue
            print(f"  完成，用时 {time.time()-t0:.1f}s  窗口云量 {quality['cloud_pct']:.1f}%  "
                  f"水体 {quality['water_pct']:.1f}%")
            return item, quality, center, data, cat
        print(f"  ✗ {CATALOG_LABEL.get(cat, cat)} 所有候选景都读失败"
              + (f"：{last_exc}" if last_exc else "") + "，换数据源")
    return None


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="抓取真实 Sentinel-2 洪涝影像")
    ap.add_argument("--event", default="all", help="事件名或 all")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "real"))
    ap.add_argument("--size", type=int, default=None, help="窗口边长（像元，10m）")
    ap.add_argument("--max-cloud", type=float, default=25.0, help="AOI 窗口最大可接受云量(%)")
    ap.add_argument("--list", action="store_true", help="列出内置事件")
    ap.add_argument("--allow-cloudy", action="store_true", help="窗口云量超标时仍强制使用（默认拒绝）")
    args = ap.parse_args()

    if args.list:
        print("内置洪涝事件：")
        for k, v in EVENTS.items():
            print(f"  {k:16s} {v['label']}")
            print(f"  {'':16s} AOI={v['aoi']}  灾前={v['pre'][:2]}  灾后={v['post'][:2]}")
        return

    keys = list(EVENTS) if args.event == "all" else [args.event]
    for k in keys:
        if k not in EVENTS:
            raise SystemExit(f"未知事件 {k}，可选：{list(EVENTS)}")

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "samples.json")
    manifest: Dict[str, Any] = {"samples": []}
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        manifest.setdefault("samples", [])

    manifest.update({
        "note": "真实 Sentinel-2 L2A 影像（AWS 公开 COG，经 Element84 Earth Search 检索），"
                "用于演示与验证；不含人工标注掩膜。",
        "pixel_size_m": 10.0,
        "bands": ["B2(blue)", "B3(green)", "B4(red)", "B8(nir)"],
    })

    for k in keys:
        entry = fetch_event(k, EVENTS[k], args.out, size=args.size, max_cloud=args.max_cloud, allow_cloudy=args.allow_cloudy)
        if entry is None:
            print(f"✗ {k} 抓取失败")
            continue
        manifest["samples"] = [s for s in manifest["samples"] if s.get("id") != k] + [entry]
        write_json_atomic(manifest_path, manifest)
        print(f"✓ {k} 已写入 {manifest_path}")

    print("\n全部完成。产物目录：", args.out)
    print("下一步：python -c \"from src import FloodDetector; "
          "print(FloodDetector().detect('data/real/dongting2024_post.tif').summary_text())\"")


if __name__ == "__main__":
    main()
