"""
慧眼识灾 · 洪灾事件注册表与检索
=================================

这个模块回答用户最常卡住的两个问题：**去哪儿看**、**填什么日期**。

两块能力：

1. **内部检索**（离线）：在内置事件库上做关键词 / 年份 / 地区 / 邻近检索，
   直接产出可以填进界面的事件模板。
2. **联网发现**（可选）：从 GDACS（全球灾害预警系统，公开 JSON、免密钥）
   拉取真实洪灾记录，转成同样格式的模板。

设计要点
--------

* **事件只记录一个日期**（洪水发生/峰值日），灾前/灾后时间窗由
  `build_template()` 按统一规则推导。这样任何来源——内置库还是联网发现——
  的事件都能立刻变成可用模板，不需要人工填四个日期（日期填错正是用户踩坑最多的地方）。
* **坐标标注精度**：`precision="aoi"` 表示已核对到具体区域，`"region"` 表示
  只有事件区域近似中心。精度不足的会在模板提示里明确要求用户在地图上微调，
  避免"下载成功但分析的是别处"。
* 联网只访问白名单主机，并阻断解析到内网/环回的地址（防线与
  `scripts/fetch_real_samples.py` 一致）。

用法
----

    from src.events import load_registry, search_events, build_template

    events = load_registry()
    hits = search_events("鄱阳湖", events=events)
    template = build_template(hits[0])
    template["aoi"], template["post"]        # 直接填进界面
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

GDACS_SEARCH_API = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
USER_AGENT = "flood-eyes/0.3 (educational; flood event registry)"

# 联网只允许访问这些主机（GDACS 官方域名）
_ALLOWED_HOSTS = ("www.gdacs.org", "gdacs.org")

# 模板默认时间窗（天）。统一在这里定义，便于调整口径。
DEFAULT_PRE_SPAN = 30   # 灾前窗口长度
DEFAULT_PRE_GAP = 15    # 灾前窗口距事件日的间隔
DEFAULT_POST_SPAN = 30  # 灾后窗口长度
DEFAULT_POST_GAP = 3    # 灾后窗口起点距事件日的间隔

# 邻近检索的默认半径（公里）
DEFAULT_NEAR_KM = 300.0

# Sentinel-2A 首景日期。本项目只做 Sentinel-2 光学识别，
# 早于该日期的洪灾没有可用影像，选中也跑不出结果——必须提前告诉用户。
SENTINEL2_START = "2015-06-23"

_EARTH_R_KM = 6371.0088


def registry_path(root: Optional[str] = None) -> str:
    """内置事件库位置（data/events/flood_events.json）。"""
    if root is None:
        # 延迟导入，避免 events 模块对 paths 产生硬依赖
        from src.paths import bundle_root

        root = bundle_root()
    return os.path.join(root, "data", "events", "flood_events.json")


# --------------------------------------------------------------------------
# 安全请求
# --------------------------------------------------------------------------


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


def assert_public_https_url(raw_url: str) -> str:
    """发请求前的统一校验：协议 + 主机白名单 + 解析后 IP 不得指向内网。

    本模块只与 GDACS 通信，白名单是固定的；校验的目的不是防"外部输入"，
    而是防止请求因配置/DNS 问题打到本机或内网服务上。
    """
    parsed = urllib.parse.urlparse(str(raw_url))
    if parsed.scheme != "https":
        raise ValueError(f"拒绝非 https 地址：{str(raw_url)[:100]}")
    host = (parsed.hostname or "").lower()
    if host not in _ALLOWED_HOSTS:
        raise ValueError(f"拒绝白名单之外的联网主机：{host or '(空主机)'}")
    import socket

    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise ValueError(f"联网主机无法解析：{host}") from exc
    for info in infos:
        addr = info[4][0]
        if _is_private_ip(addr):
            raise ValueError(f"联网主机 {host} 解析到非公网地址 {addr}，已拒绝")
    return str(raw_url)


def _http_json(url: str, *, timeout: float = 30.0) -> Any:
    """带超时与地址校验的 GET + JSON 解析。"""
    safe_url = assert_public_https_url(url)
    req = urllib.request.Request(
        safe_url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.load(resp)


# --------------------------------------------------------------------------
# 事件模型
# --------------------------------------------------------------------------


@dataclass
class FloodEvent:
    """一起洪灾事件。"""

    id: str
    label: str
    lon: float
    lat: float
    event_date: str            # 洪水发生/峰值日 YYYY-MM-DD
    region: str = ""           # 省/国家
    severity: str = ""         # 严重程度描述（人数、面积等）
    note: str = ""             # 背景说明
    source: str = "人工整理"    # 来源（机构名 / 接口维度）
    precision: str = "aoi"     # aoi=已核对到具体区域，region=仅区域近似中心
    kind: str = "curated"      # curated=内置，online=联网发现
    pre_span_days: int = DEFAULT_PRE_SPAN
    pre_gap_days: int = DEFAULT_PRE_GAP
    post_span_days: int = DEFAULT_POST_SPAN
    post_gap_days: int = DEFAULT_POST_GAP
    # 手工调过的时间窗，格式 [起, 止, 目标日]；给了就直接用，不再按规则推导。
    # 保留它是为了让项目里原有事件（作者按实际汛情调过）口径不变。
    pre: Optional[List[str]] = None
    post: Optional[List[str]] = None

    # -- 基本信息 ----------------------------------------------------------

    @property
    def year(self) -> Optional[int]:
        try:
            return int(str(self.event_date)[:4])
        except (TypeError, ValueError):
            return None

    @property
    def analysable(self) -> bool:
        """该事件是否落在 Sentinel-2 可用时段内。"""
        return str(self.event_date)[:10] >= SENTINEL2_START

    def distance_km(self, lon: float, lat: float) -> float:
        """到给定经纬度的球面距离（公里）。"""
        return haversine_km(self.lon, self.lat, lon, lat)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FloodEvent":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        data = {k: v for k, v in payload.items() if k in known}
        for required in ("id", "label", "lon", "lat", "event_date"):
            if required not in data:
                raise ValueError(f"事件缺少必填字段 {required}")
        data["lon"] = float(data["lon"])
        data["lat"] = float(data["lat"])
        data["event_date"] = str(data["event_date"])[:10]
        for key in ("pre", "post"):
            value = data.get(key)
            if value is None:
                continue
            if not (isinstance(value, (list, tuple)) and len(value) == 3):
                raise ValueError(f"{key} 需为 [起, 止, 目标日] 三个日期")
            data[key] = [str(v)[:10] for v in value]
        return cls(**data)


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """两点球面距离（公里）。"""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _EARTH_R_KM * math.asin(min(1.0, math.sqrt(a)))


# --------------------------------------------------------------------------
# 读写
# --------------------------------------------------------------------------


def load_registry(path: Optional[str] = None) -> List[FloodEvent]:
    """读取内置事件库。

    库里坏掉个别条目不影响其余：逐条 try/except 跳过并告警，
    避免一份手改出错的事件库让整个软件起不来。
    """
    target = path or registry_path()
    if not os.path.isfile(target):
        return []
    try:
        with open(target, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[warn] 事件库无法解析，按空库处理：{target}（{type(exc).__name__}）")
        return []

    raw = payload.get("events", []) if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        return []

    out: List[FloodEvent] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            out.append(FloodEvent.from_dict(item))
        except (ValueError, TypeError) as exc:
            print(f"[warn] 跳过无法解析的事件条目：{exc}")
    return out


def save_registry(events: Sequence[FloodEvent], path: Optional[str] = None,
                  root: Optional[str] = None) -> str:
    """写回事件库（原子替换，避免中途崩溃留下截断 JSON）。"""
    target = Path(path or registry_path(root))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "慧眼识灾内置洪灾事件库；event_date 为洪水发生/峰值日，"
                "灾前灾后时间窗由 src/events.py 的 build_template() 推导。",
        "events": [e.to_dict() for e in events],
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(target))
    return str(target)


# --------------------------------------------------------------------------
# 内部检索
# --------------------------------------------------------------------------


def _score_match(event: FloodEvent, query: str) -> float:
    """关键词匹配打分，0 表示不匹配。"""
    q = (query or "").strip().lower()
    if not q:
        return 0.0
    label = event.label.lower()
    region = (event.region or "").lower()
    ident = event.id.lower()
    note = (event.note or "").lower()

    if q == ident or q == label:
        return 100.0
    if q in label:
        return 60.0 + 20.0 * (len(q) / max(len(label), 1))
    if q in ident:
        return 50.0
    if q in region:
        return 40.0
    if q in note:
        return 20.0
    if q in event.event_date:
        return 15.0
    # 说明：这里刻意不做"逐字模糊匹配"。试过一版按单字命中率的兜底，
    # 结果查 "不存在的事件" 也能靠"的""在"这类常见字命中 4 条，
    # 给出看着可信、实际无关的结果。宁可不返回，也不要误导。
    # 子串匹配已能覆盖真实用法（"鄱阳湖" 命中 "2020·江西鄱阳湖特大洪水"）。
    return 0.0


def search_events(
    query: str = "",
    *,
    events: Optional[Sequence[FloodEvent]] = None,
    year: Optional[int] = None,
    region: Optional[str] = None,
    limit: int = 20,
) -> List[FloodEvent]:
    """内部搜索引擎：按关键词 / 年份 / 地区检索事件。

    query 为空时按事件日倒序返回（最新在前），可叠加 year/region 过滤。
    命中多条时按得分降序，同分按事件日新的在前。
    """
    pool = list(events) if events is not None else load_registry()

    if year is not None:
        pool = [e for e in pool if e.year == int(year)]
    if region:
        r = region.strip().lower()
        pool = [e for e in pool if r in (e.region or "").lower() or r in e.label.lower()]

    q = (query or "").strip()
    if not q:
        pool.sort(key=lambda e: e.event_date, reverse=True)
        return pool[: max(0, int(limit))]

    scored = [(s, e) for s, e in ((_score_match(e, q), e) for e in pool) if s > 0]
    # 稳定排序：先按事件日新→旧，再按得分高→低（保留前一次的日期序）
    scored.sort(key=lambda t: t[1].event_date, reverse=True)
    scored.sort(key=lambda t: -t[0])
    return [e for _, e in scored][: max(0, int(limit))]


def nearest_events(
    lon: float,
    lat: float,
    *,
    events: Optional[Sequence[FloodEvent]] = None,
    max_km: float = DEFAULT_NEAR_KM,
    limit: int = 5,
) -> List[Tuple[float, FloodEvent]]:
    """邻近检索：给定坐标，找最近的若干起洪灾事件。

    用于回答"我点的这个地方历史上有没有洪灾、什么时间"——
    这正是用户随便点一个位置却不知道填什么日期的场景。
    """
    pool = list(events) if events is not None else load_registry()
    ranked = sorted(((e.distance_km(lon, lat), e) for e in pool), key=lambda t: t[0])
    return [(d, e) for d, e in ranked if d <= float(max_km)][: max(0, int(limit))]


# --------------------------------------------------------------------------
# 模板生成
# --------------------------------------------------------------------------


def _shift_date(day: str, days: int) -> str:
    return (_dt.date.fromisoformat(str(day)[:10]) + _dt.timedelta(days=int(days))).isoformat()


def build_template(
    event: FloodEvent,
    *,
    size: int = 1280,
    pre_span: Optional[int] = None,
    pre_gap: Optional[int] = None,
    post_span: Optional[int] = None,
    post_gap: Optional[int] = None,
) -> Dict[str, Any]:
    """把事件转成可直接填进界面的模板。

    时间窗推导规则（以事件日为 T）：
        灾前 = [T - pre_gap - pre_span, T - pre_gap]，目标日 T - pre_gap
        灾后 = [T + post_gap, T + post_gap + post_span]，目标日 T + post_gap
    默认 pre_gap=15 / pre_span=30 / post_gap=3 / post_span=30：
    灾前取事件前 45~15 天（汛前基线），灾后取事件后 3~33 天（含峰值与退水）。

    返回 dict 同时含界面字段（label/aoi/pre/post/size/note）与元信息（_event）。
    """
    if not event.event_date:
        raise ValueError(f"事件 {event.id} 缺少日期，无法生成模板")
    ps = event.pre_span_days if pre_span is None else int(pre_span)
    pg = event.pre_gap_days if pre_gap is None else int(pre_gap)
    qs = event.post_span_days if post_span is None else int(post_span)
    qg = event.post_gap_days if post_gap is None else int(post_gap)

    # 事件自带手工时间窗时优先使用（保持原有口径），否则按规则推导
    pre = tuple(event.pre) if event.pre else (
        _shift_date(event.event_date, -(pg + ps)),
        _shift_date(event.event_date, -pg),
        _shift_date(event.event_date, -pg),
    )
    post = tuple(event.post) if event.post else (
        _shift_date(event.event_date, qg),
        _shift_date(event.event_date, qg + qs),
        _shift_date(event.event_date, qg),
    )
    pre_start, pre_end = pre[0], pre[1]
    post_start, post_end = post[0], post[1]

    return {
        "label": event.label,
        "aoi": (float(event.lon), float(event.lat)),
        "pre": (pre_start, pre_end, pre[2]),
        "post": (post_start, post_end, post[2]),
        "size": int(size),
        "note": event.note,
        "event_date": event.event_date,
        "source": event.source,
        "precision": event.precision,
        "region": event.region,
        "analysable": event.analysable,
        "_event": event.to_dict(),
    }


def templates_for(events: Optional[Sequence[FloodEvent]] = None) -> Dict[str, Dict[str, Any]]:
    """{id: 模板} —— 与 fetch_real_samples.EVENTS 同构，可直接替换使用。"""
    pool = list(events) if events is not None else load_registry()
    return {e.id: build_template(e) for e in pool}


def template_hint(event: FloodEvent, distance_km: Optional[float] = None) -> str:
    """给界面用的一行提示文案。"""
    bits = [f"**{event.label}**"]
    if event.region:
        bits.append(f"地区：{event.region}")
    bits.append(f"事件日：{event.event_date}")
    if event.severity:
        bits.append(f"规模：{event.severity}")
    if distance_km is not None:
        bits.append(f"距所选点约 {distance_km:.0f} km")
    precision = "已核对到具体区域" if event.precision == "aoi" else "仅事件区域近似中心，请在地图上微调"
    bits.append(f"坐标精度：{precision}")
    bits.append(f"来源：{event.source}")
    if event.note:
        bits.append(f"背景：{event.note}")
    return " ｜ ".join(bits)


# --------------------------------------------------------------------------
# 联网发现（GDACS）
# --------------------------------------------------------------------------


def fetch_gdacs_floods(
    from_date: str,
    to_date: str,
    *,
    countries: Optional[Sequence[str]] = None,
    timeout: float = 30.0,
) -> List[FloodEvent]:
    """从 GDACS 拉取洪灾事件（FL）并转成 FloodEvent 列表。

    GDACS 公开免密钥，但坐标是**事件区域近似中心**，
    因此转出来的事件 precision="region"，界面会提示用户在地图上微调。

    countries 可传 ["China"] 之类的国家名做过滤。
    """
    query = urllib.parse.urlencode({
        "eventlist": "FL",
        "fromDate": str(from_date)[:10],
        "toDate": str(to_date)[:10],
    })
    payload = _http_json(f"{GDACS_SEARCH_API}?{query}", timeout=timeout)
    feats = payload.get("features", []) if isinstance(payload, dict) else []
    if not isinstance(feats, list):
        return []

    wanted = {str(c).strip().lower() for c in (countries or []) if str(c).strip()}
    out: List[FloodEvent] = []
    for feat in feats:
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties") or {}
        coords = (feat.get("geometry") or {}).get("coordinates")
        if not (isinstance(coords, (list, tuple)) and len(coords) >= 2):
            continue
        try:
            lon, lat = float(coords[0]), float(coords[1])
        except (TypeError, ValueError):
            continue
        country = str(props.get("country") or "").strip()
        if wanted and country.lower() not in wanted:
            continue
        date = str(props.get("fromdate") or "")[:10]
        if len(date) != 10:
            continue
        event_id = props.get("eventid") or props.get("eventid_") or ""
        name = str(props.get("eventname") or "").strip() or "洪灾"
        alert = str(props.get("alertlevel") or "").strip()
        out.append(FloodEvent(
            id=f"gdacs{event_id}" if event_id else f"gdacs-{date}-{lon:.2f}-{lat:.2f}",
            label=f"{date[:4]}·{country} {name}".strip(),
            lon=lon,
            lat=lat,
            event_date=date,
            region=country,
            severity=f"GDACS 预警等级 {alert}" if alert else "",
            note=f"GDACS 收录的洪灾事件（事件编号 {event_id or '未知'}）。"
                 "坐标为事件区域近似中心，建议在地图上确认具体受淹区后再分析。",
            source="GDACS（全球灾害预警系统）",
            precision="region",
            kind="online",
        ))
    return out


def merge_registry(
    new_events: Iterable[FloodEvent],
    *,
    path: Optional[str] = None,
    root: Optional[str] = None,
    overwrite: bool = False,
) -> Tuple[int, int]:
    """把新事件并入事件库，按 id 去重。返回 (新增数, 更新数)。"""
    existing = load_registry(path or registry_path(root))
    index = {e.id: e for e in existing}
    added = updated = 0
    for event in new_events:
        if event.id in index:
            if overwrite:
                index[event.id] = event
                updated += 1
            continue
        index[event.id] = event
        added += 1
    save_registry(list(index.values()), path=path, root=root)
    return added, updated


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------


def validate_event(event: FloodEvent) -> List[str]:
    """返回该事件的问题列表（空表示没问题）。"""
    problems: List[str] = []
    if not event.id:
        problems.append("缺少 id")
    if not event.label:
        problems.append("缺少 label")
    if not (-180.0 <= event.lon <= 180.0):
        problems.append(f"经度越界：{event.lon}")
    if not (-90.0 <= event.lat <= 90.0):
        problems.append(f"纬度越界：{event.lat}")
    day = str(event.event_date or "")
    if len(day) != 10 or day[4] != "-" or day[7] != "-":
        problems.append(f"事件日格式应为 YYYY-MM-DD：{day}")
    else:
        try:
            _dt.date.fromisoformat(day)
        except ValueError:
            problems.append(f"事件日不是合法日期：{day}")
    if event.precision not in ("aoi", "region"):
        problems.append(f"precision 只能是 aoi 或 region：{event.precision}")
    if not event.analysable:
        problems.append(f"早于 Sentinel-2 可用日期（{SENTINEL2_START}），本项目取不到影像、无法分析")
    for name in ("pre_span_days", "pre_gap_days", "post_span_days", "post_gap_days"):
        value = getattr(event, name)
        if not isinstance(value, int) or value < 0:
            problems.append(f"{name} 应为非负整数：{value!r}")
    for name in ("pre", "post"):
        window = getattr(event, name)
        if window is None:
            continue
        if not (isinstance(window, (list, tuple)) and len(window) == 3):
            problems.append(f"{name} 需为 [起, 止, 目标日] 三个日期：{window!r}")
            continue
        if window[0] > window[1]:
            problems.append(f"{name} 起始日晚于结束日：{window[0]} > {window[1]}")
    return problems


def validate_registry(events: Optional[Sequence[FloodEvent]] = None) -> Dict[str, List[str]]:
    """校验整库，返回 {事件id: 问题列表}（只含有问题的事件）。"""
    pool = list(events) if events is not None else load_registry()
    counts: Dict[str, int] = {}
    out: Dict[str, List[str]] = {}
    for event in pool:
        counts[event.id] = counts.get(event.id, 0) + 1
        problems = validate_event(event)
        if problems:
            out[event.id] = problems
    for ident, count in counts.items():
        if count > 1:
            out.setdefault(ident, []).append(f"id 重复出现 {count} 次")
    return out
