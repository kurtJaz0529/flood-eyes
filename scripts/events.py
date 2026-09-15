"""
慧眼识灾 · 洪灾事件检索与模板生成（命令行）
=============================================

用法：

    # 列出内置事件库
    python scripts/events.py --list

    # 关键词 / 年份 / 地区检索
    python scripts/events.py --search 鄱阳湖
    python scripts/events.py --search 洪涝 --year 2024
    python scripts/events.py --region 湖南

    # 邻近检索：这个坐标附近历史上有什么洪灾？
    python scripts/events.py --near 116.92,29.55

    # 看某个事件，并打印它生成的填空模板
    python scripts/events.py --show poyang2020

    # 导出全部模板（JSON，可直接替换 fetch_real_samples.EVENTS）
    python scripts/events.py --export outputs/events_templates.json

    # 联网发现（GDACS，公开接口免密钥）
    python scripts/events.py --online --from 2024-06-01 --to 2024-12-31
    python scripts/events.py --online --from 2024-06-01 --to 2024-12-31 --country China --save

    # 校验事件库
    python scripts/events.py --validate
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

from src.events import (  # noqa: E402
    FloodEvent,
    build_template,
    fetch_gdacs_floods,
    load_registry,
    merge_registry,
    nearest_events,
    registry_path,
    save_registry,
    search_events,
    templates_for,
    validate_registry,
)


def _fmt_event(event: FloodEvent, prefix: str = "  ") -> str:
    tags = []
    if event.region:
        tags.append(event.region)
    tags.append(event.precision)
    if event.kind == "online":
        tags.append("联网")
    return f"{prefix}{event.event_date}  {event.label}  [{'/'.join(tags)}]  ({event.lon:.3f}, {event.lat:.3f})"


def _print_template(event: FloodEvent) -> None:
    tpl = build_template(event)
    pre, post = tpl["pre"], tpl["post"]
    print(f"  模板 id      : {event.id}")
    print(f"  名称         : {tpl['label']}")
    print(f"  经度 / 纬度  : {tpl['aoi'][0]:.4f} / {tpl['aoi'][1]:.4f}")
    print(f"  灾前开始     : {pre[0]}")
    print(f"  灾前结束     : {pre[1]}")
    print(f"  灾后开始     : {post[0]}")
    print(f"  灾后结束     : {post[1]}")
    print(f"  窗口边长     : {tpl['size']}")
    print(f"  可否分析     : {'是' if tpl['analysable'] else '否（早于 Sentinel-2 可用日期）'}")
    print(f"  坐标精度     : {tpl['precision']}")
    print(f"  来源         : {tpl['source']}")
    if tpl["note"]:
        print(f"  背景         : {tpl['note']}")


def cmd_list(events: Sequence[FloodEvent]) -> int:
    print(f"内置事件库：{registry_path()}")
    print(f"共 {len(events)} 起")
    print("-" * 72)
    for e in search_events(events=events, limit=10_000):
        print(_fmt_event(e, prefix=""))
    return 0


def cmd_search(args: argparse.Namespace, events: Sequence[FloodEvent]) -> int:
    hits = search_events(
        args.search or "",
        events=events,
        year=args.year,
        region=args.region,
        limit=args.limit,
    )
    cond = []
    if args.search:
        cond.append(f"关键词「{args.search}」")
    if args.year:
        cond.append(f"年份 {args.year}")
    if args.region:
        cond.append(f"地区「{args.region}」")
    print("检索条件：" + ("、".join(cond) if cond else "（无，按事件日倒序）"))
    print(f"命中 {len(hits)} 起")
    print("-" * 72)
    for e in hits:
        print(_fmt_event(e, prefix=""))
    if not hits:
        print("  没有命中。可用 --list 看全部，或用 --online 联网发现新事件。")
    return 0


def cmd_near(args: argparse.Namespace, events: Sequence[FloodEvent]) -> int:
    raw = str(args.near)
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise SystemExit(f"--near 需要“经度,纬度”两个值，收到：{raw!r}")
    try:
        lon, lat = float(parts[0]), float(parts[1])
    except ValueError:
        raise SystemExit(f"--near 必须是数字，收到：{raw!r}") from None
    if not (-180.0 <= lon <= 180.0) or not (-90.0 <= lat <= 90.0):
        raise SystemExit(f"--near 经纬度超出范围：{lon}, {lat}")

    hits = nearest_events(lon, lat, events=events, max_km=args.max_km, limit=args.limit)
    print(f"坐标 ({lon:.4f}, {lat:.4f}) 周边 {args.max_km:.0f} km 内的已知洪灾事件：")
    print("-" * 72)
    if not hits:
        print("  没有找到已知事件。可以：")
        print("    · 直接在地图上确认该地是否有水体需要提取；")
        print("    · 放宽半径：--max-km 1000；")
        print("    · 联网发现新事件：--online --from ... --to ...")
        return 0
    for dist, e in hits:
        print(f"  {dist:7.1f} km  {e.event_date}  {e.label}")
    nearest_dist, nearest = hits[0]
    print("-" * 72)
    print("最近事件生成的填空模板：")
    _print_template(nearest)
    return 0


def cmd_show(args: argparse.Namespace, events: Sequence[FloodEvent]) -> int:
    target = str(args.show)
    found = [e for e in events if e.id == target]
    if not found:
        found = search_events(target, events=events, limit=1)
    if not found:
        raise SystemExit(f"未找到事件：{target}")
    event = found[0]
    print(_fmt_event(event, prefix=""))
    print(f"  规模 : {event.severity or '-'}")
    print(f"  来源 : {event.source}")
    print("-" * 72)
    _print_template(event)
    return 0


def cmd_export(args: argparse.Namespace, events: Sequence[FloodEvent]) -> int:
    tpl = templates_for(events)
    target = Path(str(args.export))
    target.parent.mkdir(parents=True, exist_ok=True)
    # 去掉 _event 冗余字段，只留界面需要的部分
    payload = {
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "note": "事件填空模板：aoi=(经度,纬度)，pre/post=(起,止,目标日)，size=窗口边长",
        "templates": {k: {kk: vv for kk, vv in v.items() if kk != "_event"} for k, v in tpl.items()},
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已导出 {len(tpl)} 个模板 -> {target}")
    return 0


def cmd_online(args: argparse.Namespace, events: Sequence[FloodEvent]) -> int:
    from_date = str(args.from_date or "")
    to_date = str(args.to or "")
    if not from_date or not to_date:
        raise SystemExit("--online 需要同时给 --from 与 --to（YYYY-MM-DD）")

    countries: Optional[List[str]] = None
    if args.country:
        countries = [c.strip() for c in str(args.country).split(",") if c.strip()]

    print(f"从 GDACS 拉取洪灾事件：{from_date} ~ {to_date}"
          + (f"  国家过滤：{countries}" if countries else ""))
    try:
        found = fetch_gdacs_floods(from_date, to_date, countries=countries)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"联网发现失败：{type(exc).__name__}: {exc}\n"
                         "（GDACS 需要可访问外网；离线时可用 --list 检索内置事件库）") from exc

    print(f"发现 {len(found)} 起")
    print("-" * 72)
    for e in found:
        print(_fmt_event(e, prefix=""))

    if not found:
        return 0

    if args.save:
        added, updated = merge_registry(found, overwrite=bool(args.overwrite))
        print("-" * 72)
        print(f"已并入事件库：新增 {added} 起，更新 {updated} 起 -> {registry_path()}")
        print("提示：GDACS 坐标是事件区域近似中心，用前请在地图上确认具体受淹区。")
    else:
        print("-" * 72)
        print("（未写入事件库；加 --save 可并入，加 --overwrite 可更新同 id 条目）")
    return 0


def cmd_validate(events: Sequence[FloodEvent]) -> int:
    problems = validate_registry(events)
    print(f"校验事件库：{registry_path()}")
    print(f"共 {len(events)} 起")
    if not problems:
        print("✓ 全部通过")
        return 0
    print(f"✗ {len(problems)} 起事件有问题：")
    for ident, items in problems.items():
        print(f"  {ident}:")
        for item in items:
            print(f"    - {item}")
    return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="慧眼识灾 · 洪灾事件检索与模板生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--list", action="store_true", help="列出全部内置事件")
    ap.add_argument("--search", default=None, help="关键词检索（名称/地区/编号/备注/日期）")
    ap.add_argument("--year", type=int, default=None, help="按年份过滤")
    ap.add_argument("--region", default=None, help="按地区过滤")
    ap.add_argument("--near", default=None, help="邻近检索，格式：经度,纬度")
    ap.add_argument("--max-km", type=float, default=300.0, help="邻近检索半径（公里，默认 300）")
    ap.add_argument("--show", default=None, help="显示某个事件及其模板（事件 id 或关键词）")
    ap.add_argument("--export", default=None, help="导出全部填空模板到 JSON")
    ap.add_argument("--online", action="store_true", help="联网发现洪灾事件（GDACS）")
    ap.add_argument("--from", dest="from_date", default=None,
                    help="联网发现的起始日期 YYYY-MM-DD")
    ap.add_argument("--to", default=None, help="联网发现的结束日期 YYYY-MM-DD")
    ap.add_argument("--country", default=None, help="联网发现的国家过滤，多个用逗号分隔")
    ap.add_argument("--save", action="store_true", help="把联网发现的事件并入事件库")
    ap.add_argument("--overwrite", action="store_true", help="并入时覆盖同 id 条目")
    ap.add_argument("--validate", action="store_true", help="校验事件库")
    ap.add_argument("--limit", type=int, default=30, help="最多显示条数")
    args = ap.parse_args(argv)

    events = load_registry()

    if args.validate:
        return cmd_validate(events)
    if args.online:
        return cmd_online(args, events)
    if args.near:
        return cmd_near(args, events)
    if args.show:
        return cmd_show(args, events)
    if args.export:
        return cmd_export(args, events)
    if args.search is not None or args.year or args.region:
        return cmd_search(args, events)
    if args.list:
        return cmd_list(events)

    ap.print_help()
    print(f"\n当前事件库：{registry_path()}（{len(events)} 起）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
