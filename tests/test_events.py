# -*- coding: utf-8 -*-
"""事件库与检索的回归测试。

运行：
    python tests/test_events.py
    python -m pytest tests/test_events.py -q

覆盖 `src/events.py`：事件模型、内部检索、邻近检索、模板生成、校验、读写往返。

约定：每条用例都对应一处**真实行为或已修缺陷**，注释写明意图。
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

from src import events as ev  # noqa: E402


def _event(**over) -> ev.FloodEvent:
    base = dict(id="t1", label="测试事件", lon=116.0, lat=29.0, event_date="2024-07-01")
    base.update(over)
    return ev.FloodEvent(**base)


class TestRegistry(unittest.TestCase):
    def test_builtin_registry_is_valid(self):
        lib = ev.load_registry()
        self.assertGreaterEqual(len(lib), 10, "内置事件库条目过少")
        self.assertEqual(ev.validate_registry(lib), {}, "内置事件库存在非法条目")

    def test_ids_are_unique(self):
        ids = [e.id for e in ev.load_registry()]
        self.assertEqual(len(ids), len(set(ids)), "事件 id 必须唯一")

    def test_all_events_analysable(self):
        """内置事件都应落在 Sentinel-2 可用时段内，否则选中也跑不出结果。"""
        bad = [e.id for e in ev.load_registry() if not e.analysable]
        self.assertEqual(bad, [], f"这些事件早于 Sentinel-2 可用日期，不应入库：{bad}")

    def test_preserved_events_keep_hand_tuned_windows(self):
        """项目原有事件的手工时间窗必须原样保留（作者按实际汛情调过）。"""
        lib = {e.id: e for e in ev.load_registry()}
        py = lib["poyang2020"]
        self.assertEqual(py.pre, ["2020-05-08", "2020-06-05", "2020-05-20"])
        self.assertEqual(py.post, ["2020-07-10", "2020-07-30", "2020-07-13"])
        zz = lib["zhuozhou2023"]
        self.assertEqual(zz.pre, ["2023-07-08", "2023-07-28", "2023-07-18"])

    def test_cache_ids_present(self):
        """带本地缓存影像的事件 id 必须保留，否则缓存命不中。"""
        ids = {e.id for e in ev.load_registry()}
        for key in ("poyang2020", "zhuozhou2023", "brazil2024"):
            self.assertIn(key, ids, f"{key} 缺失会导致本地缓存失效")


class TestSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = ev.load_registry()

    def test_keyword_hits_expected_event(self):
        hits = ev.search_events("鄱阳湖", events=self.lib)
        self.assertTrue(hits)
        self.assertEqual(hits[0].id, "poyang2020")

    def test_region_keyword(self):
        hits = ev.search_events("湖南", events=self.lib)
        self.assertEqual({e.region for e in hits}, {"湖南"})

    def test_year_query_returns_only_that_year(self):
        """回归：早期实现带"逐字模糊匹配"兜底，查 2024 会把 2020/2021 的事件也捞回来。"""
        hits = ev.search_events("2024", events=self.lib)
        self.assertTrue(hits)
        self.assertEqual({e.year for e in hits}, {2024})
        self.assertEqual(len(hits), 7)

    def test_nonsense_query_returns_nothing(self):
        """乱查询必须返回空——宁可没有结果，也不要给出看着可信的无关事件。

        回归：逐字兜底曾让"不存在的事件"靠"的""在"这类常见字命中 4 条。
        """
        for query in ("不存在的事件", "xyz", "qwerty", "2024年火星洪水"):
            self.assertEqual(ev.search_events(query, events=self.lib), [],
                             f"{query!r} 不应命中任何事件")

    def test_empty_query_returns_all_newest_first(self):
        hits = ev.search_events(events=self.lib, limit=1000)
        self.assertEqual(len(hits), len(self.lib))
        dates = [e.event_date for e in hits]
        self.assertEqual(dates, sorted(dates, reverse=True), "应按事件日倒序")

    def test_limit_is_respected(self):
        self.assertEqual(len(ev.search_events(events=self.lib, limit=3)), 3)


class TestNearest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = ev.load_registry()

    def test_zero_distance_at_event_centre(self):
        hits = ev.nearest_events(116.30, 29.15, events=self.lib, max_km=1.0)
        self.assertTrue(hits)
        dist, event = hits[0]
        self.assertEqual(event.id, "poyang2020")
        self.assertLess(dist, 1.0)

    def test_reports_approximate_distance(self):
        """用户实际点过的位置：距鄱阳湖事件约 75 km，应能被检索到。"""
        hits = ev.nearest_events(116.92, 29.55, events=self.lib, max_km=300.0)
        self.assertTrue(hits)
        dist, event = hits[0]
        self.assertEqual(event.id, "poyang2020")
        self.assertGreater(dist, 50.0)
        self.assertLess(dist, 100.0)

    def test_max_km_filters_out_far_events(self):
        self.assertEqual(ev.nearest_events(0.0, 0.0, events=self.lib, max_km=10.0), [])

    def test_sorted_by_distance(self):
        hits = ev.nearest_events(116.9, 29.5, events=self.lib, max_km=2000.0, limit=5)
        dists = [d for d, _ in hits]
        self.assertEqual(dists, sorted(dists))

    def test_haversine_known_distance(self):
        # 北京(116.40,39.90) 到 上海(121.47,31.23) 约 1060~1080 km
        d = ev.haversine_km(116.40, 39.90, 121.47, 31.23)
        self.assertGreater(d, 1000.0)
        self.assertLess(d, 1150.0)


class TestTemplate(unittest.TestCase):
    def test_derived_window_when_event_has_no_explicit_dates(self):
        """未给 pre/post 的事件按规则推导：灾前 T-45~T-15，灾后 T+3~T+33。"""
        tpl = ev.build_template(_event(event_date="2024-07-01"))
        self.assertEqual(tpl["pre"][0], "2024-05-17")
        self.assertEqual(tpl["pre"][1], "2024-06-16")
        self.assertEqual(tpl["post"][0], "2024-07-04")
        self.assertEqual(tpl["post"][1], "2024-08-03")

    def test_explicit_window_wins(self):
        tpl = ev.build_template(_event(pre=["2024-01-01", "2024-01-31", "2024-01-15"],
                                       post=["2024-07-10", "2024-07-30", "2024-07-13"]))
        self.assertEqual(tpl["pre"], ("2024-01-01", "2024-01-31", "2024-01-15"))
        self.assertEqual(tpl["post"], ("2024-07-10", "2024-07-30", "2024-07-13"))

    def test_builtin_template_is_interface_ready(self):
        """模板必须能直接喂给现有界面/抓取链路（字段名与结构一致）。"""
        event = next(e for e in ev.load_registry() if e.id == "dongting2024")
        tpl = ev.build_template(event)
        for key in ("label", "aoi", "pre", "post", "size", "note"):
            self.assertIn(key, tpl)
        self.assertEqual(len(tpl["pre"]), 3, "pre 应为 (起, 止, 目标日)")
        self.assertEqual(len(tpl["post"]), 3)
        self.assertIsInstance(tpl["size"], int)
        self.assertEqual(tpl["aoi"], (112.66, 29.36))

    def test_custom_window_parameters(self):
        tpl = ev.build_template(_event(event_date="2024-07-01"),
                                pre_span=10, pre_gap=5, post_span=20, post_gap=1)
        self.assertEqual(tpl["pre"][0], "2024-06-16")
        self.assertEqual(tpl["pre"][1], "2024-06-26")
        self.assertEqual(tpl["post"][0], "2024-07-02")
        self.assertEqual(tpl["post"][1], "2024-07-22")

    def test_templates_for_covers_registry(self):
        lib = ev.load_registry()
        tpls = ev.templates_for(lib)
        self.assertEqual(set(tpls), {e.id for e in lib})

    def test_hint_mentions_key_facts(self):
        event = next(e for e in ev.load_registry() if e.id == "poyang2020")
        text = ev.template_hint(event, distance_km=75.0)
        self.assertIn(event.label, text)
        self.assertIn(event.event_date, text)
        self.assertIn("75", text)


class TestValidation(unittest.TestCase):
    def test_rejects_impossible_coordinates(self):
        self.assertTrue(ev.validate_event(_event(lon=200.0)))
        self.assertTrue(ev.validate_event(_event(lat=-95.0)))

    def test_rejects_bad_date(self):
        self.assertTrue(ev.validate_event(_event(event_date="2024/07/01")))
        self.assertTrue(ev.validate_event(_event(event_date="2024-13-45")))
        self.assertTrue(ev.validate_event(_event(event_date="")))
        self.assertEqual(ev.validate_event(_event(event_date="2024-07-01")), [])

    def test_flags_events_before_sentinel2(self):
        """Sentinel-2 只有 2015-06-23 之后的数据；早于此的事件取不到影像。"""
        old = _event(event_date="2011-10-10")
        self.assertFalse(old.analysable)
        self.assertTrue(any("Sentinel-2" in p for p in ev.validate_event(old)))

    def test_rejects_bad_precision_and_window(self):
        self.assertTrue(ev.validate_event(_event(precision="guess")))
        self.assertTrue(ev.validate_event(_event(pre=["2024-01-01"])))
        self.assertTrue(ev.validate_event(_event(pre=["2024-02-01", "2024-01-01", "2024-01-15"])))
        self.assertTrue(ev.validate_event(_event(pre_gap_days=-1)))

    def test_duplicate_ids_are_reported(self):
        dup = [_event(id="same"), _event(id="same")]
        self.assertIn("same", ev.validate_registry(dup))


class TestModelAndIO(unittest.TestCase):
    def test_from_dict_requires_core_fields(self):
        for missing in ("id", "label", "lon", "lat", "event_date"):
            payload = dict(id="x", label="y", lon=1.0, lat=2.0, event_date="2024-01-01")
            payload.pop(missing)
            with self.assertRaises(ValueError, msg=f"缺 {missing} 应报错"):
                ev.FloodEvent.from_dict(payload)

    def test_from_dict_coerces_types(self):
        event = ev.FloodEvent.from_dict(
            {"id": "x", "label": "y", "lon": "116.3", "lat": "29.15",
             "event_date": "2024-07-01T00:00:00Z", "pre": ("2024-01-01", "2024-02-01", "2024-01-15")}
        )
        self.assertEqual(event.lon, 116.3)
        self.assertEqual(event.event_date, "2024-07-01")
        self.assertEqual(event.pre, ["2024-01-01", "2024-02-01", "2024-01-15"])

    def test_from_dict_rejects_malformed_window(self):
        with self.assertRaises(ValueError):
            ev.FloodEvent.from_dict(
                {"id": "x", "label": "y", "lon": 1.0, "lat": 2.0,
                 "event_date": "2024-07-01", "post": ["2024-01-01"]}
            )

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "events.json"
            original = [
                _event(id="a", pre=["2024-01-01", "2024-02-01", "2024-01-15"]),
                _event(id="b", lon=-51.31, lat=-30.02, region="巴西"),
            ]
            ev.save_registry(original, path=str(target))
            self.assertTrue(target.is_file())

            back = ev.load_registry(str(target))
            self.assertEqual([e.id for e in back], ["a", "b"])
            self.assertEqual(back[0].pre, ["2024-01-01", "2024-02-01", "2024-01-15"])
            self.assertEqual(back[1].lon, -51.31)
            # 原子写不应留下临时文件
            self.assertEqual([p.name for p in Path(tmp).iterdir() if p.name.endswith(".tmp")], [])

    def test_load_missing_file_returns_empty(self):
        self.assertEqual(ev.load_registry(os.path.join(LOGS_MISSING, "nope.json")), [])

    def test_load_skips_corrupt_entries(self):
        """坏掉个别条目不应让整个事件库不可用。"""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "events.json"
            target.write_text(
                '{"events": [{"id":"ok","label":"好","lon":116,"lat":29,"event_date":"2024-07-01"},'
                '{"label":"缺 id"}, "junk", {"id":"bad","label":"坏","lon":"abc","lat":29,'
                '"event_date":"2024-07-01"}]}',
                encoding="utf-8",
            )
            lib = ev.load_registry(str(target))
            self.assertEqual([e.id for e in lib], ["ok"])

    def test_load_tolerates_broken_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "events.json"
            target.write_text("{ this is not json", encoding="utf-8")
            self.assertEqual(ev.load_registry(str(target)), [])


LOGS_MISSING = os.path.join(tempfile.gettempdir(), "huiyan_no_such_dir")


class TestOnlineSourceGuards(unittest.TestCase):
    def test_only_gdacs_hosts_allowed(self):
        """联网只允许 GDACS 官方域名，其余一律拒绝。"""
        for bad in ("https://evil.com/x", "http://www.gdacs.org/x",
                    "https://127.0.0.1/x", "file:///etc/passwd",
                    "https://gdacs.org.evil.com/x"):
            with self.assertRaises(ValueError, msg=f"{bad!r} 应被拒绝"):
                ev.assert_public_https_url(bad)

    def test_private_ip_detection(self):
        for ip in ("127.0.0.1", "10.0.0.1", "192.168.1.1", "169.254.169.254", "not-an-ip"):
            self.assertTrue(ev._is_private_ip(ip), f"{ip} 应判为不可信")
        self.assertFalse(ev._is_private_ip("8.8.8.8"))

    def test_fetch_rejects_unknown_host_before_network(self):
        """地址校验必须在发请求之前失败——不应真的去连外部主机。"""
        with self.assertRaises(ValueError):
            ev._http_json("https://evil.example.com/api")
        # GDACS 域名不在白名单外，但网络不可达时也不该崩到调用方之外
        self.assertIn("www.gdacs.org", ev._ALLOWED_HOSTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
