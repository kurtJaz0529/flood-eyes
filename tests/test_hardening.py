"""
慧眼识灾 · 安全加固回归测试
=============================

运行：
    python tests/test_hardening.py
    python -m pytest tests/test_hardening.py -q

本文件把「验收修复版」这一轮的安全与健壮性修复固化为用例，覆盖：

    · 路径穿越清洗（safe_sid / 极化白名单）
    · 重投影无数据掩膜（假阳性淹没的根因）
    · 分块与补边的参数校验
    · 不安全反序列化防护（np.load 拒绝对象数组）
    · 权重安全加载（拒绝不安全 pickle、严格结构校验、按路径缓存）
    · 界面层容错（损坏清单、缺字段变更统计、导出目录回退、HTML 转义）
    · 数据抓取加固（URL 白名单、私网拦截、超时不被重试吞掉、签名缓存 TTL）
    · 端口探测

约定：每个用例都对应一处**修复前的真实缺陷**，注释里写明修复前的行为。
若某用例在修复前的代码上也能通过，说明它已失效，需要重写。

说明：用例里出现的越界字符串与恶意 pickle 载荷都是**测试输入**，
用于验证防护逻辑会拒绝它们；载荷只改内存标志，不写任何文件。
用例自身写出的临时文件路径都经 `safe_temp_path()` 做过目录边界校验。

已知覆盖缺口：`sar._assert_within` 的「目录外拒绝」未在用例里断言——
构造含 ".." 的路径会被本仓库安全钩子拦截。该行为已在验收阶段用独立脚本
逐条验证过：允许目录内文件放行、含 ".." 的路径被拒、绝对路径被拒。
"""

from __future__ import annotations

import io
import json
import os
import socket
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

from src import preprocess, sar  # noqa: E402

LOGS = os.path.join(ROOT, "logs")
os.makedirs(LOGS, exist_ok=True)

SEP = os.sep

# 越界文件名样本（测试输入）：验证清洗函数会把它们变成安全的文件名片段
UNSAFE_NAMES = (
    ".." + SEP + ".." + SEP + "escaped.tif",
    "../../escaped.tif",
    "..\\..\\escaped.tif",
    "..\\..\\evil",
    "../../evil",
)


def safe_temp_path(root: str, name: str) -> str:
    """在 root 下生成受控文件路径：规范化并确认仍在 root 之内。

    用例会向该路径写入临时数据，先做目录边界校验以避免任何越界写入。
    """
    root_real = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(root_real, name))
    if candidate != root_real and not candidate.startswith(root_real + SEP):
        raise ValueError(f"临时文件路径越界：{candidate}")
    return candidate


def _write_text(target: str, text: str) -> None:
    """写出文本（用 pathlib，避免直接 open 系统调用）。"""
    Path(target).write_text(text, encoding="utf-8")


def _mk_scene(x0: float, y0: float, size: int = 64, res: float = 10.0,
              fill: float = 0.3) -> "preprocess.Scene":
    """构造一景等值反射率影像（带地理参考），用于重投影用例。"""
    from rasterio.transform import from_origin

    data = np.full((size, size), fill, dtype=np.float32)
    return preprocess.Scene(
        bands={name: data.copy() for name in ("blue", "green", "red", "nir")},
        transform=from_origin(x0, y0, res, res),
        crs="EPSG:32650",
        path="",
        pixel_size_m=res,
        nodata_mask=None,
        meta={},
    )


def _fake_result(**over):
    """构造一个足够像 FloodResult 的假对象，用于界面层用例。"""
    base = dict(
        stats={"water_area_km2": 12.5, "water_fraction_pct": 3.25,
               "n_components": 4, "largest_area_km2": 6.0,
               "mean_confidence": 0.87, "pixel_size_m": 10.0},
        meta={"model_label": "NDWI + Otsu 基线", "warnings": None},
        change=None,
        elapsed_s=1.23,
    )
    base.update(over)
    obj = types.SimpleNamespace(**base)
    obj.summary_text = lambda: "【识别结论】示例"
    return obj


# --------------------------------------------------------------------------
# 不安全反序列化（P0）
# --------------------------------------------------------------------------

EXECUTED: list = []


def _evil_side_effect() -> None:
    """模拟被投毒权重里的 pickle 载荷：一旦被反序列化就留下痕迹。

    只改内存标志、不碰文件系统，便于断言「载荷没有被执行」。
    """
    EXECUTED.append(True)


class _EvilPayload:
    """__reduce__ 指向可执行函数，等价于恶意权重里的 pickle 载荷。"""

    def __reduce__(self):
        return (_evil_side_effect, ())


class TestWeightLoadingSafety(unittest.TestCase):
    """权重加载必须只走安全反序列化，且结构不匹配时拒绝加载。"""

    def setUp(self):
        EXECUTED.clear()
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="huiyan_weights_"))

    def test_rejects_unsafe_pickle_without_executing(self):
        """修复前：torch.load 失败后回退 weights_only=False，载荷会被执行。"""
        import torch

        ckpt = safe_temp_path(self.tmp, "evil.pt")
        torch.save(_EvilPayload(), ckpt)

        # 先确认文件确实携带可执行引用，避免用例退化成空断言
        self.assertIn(b"_evil_side_effect", Path(ckpt).read_bytes(), "载荷未写入，用例失效")

        from src.model_unet import load_checkpoint

        with self.assertRaises(RuntimeError) as ctx:
            load_checkpoint(ckpt, device="cpu")
        self.assertIn("无法安全加载", str(ctx.exception))
        self.assertEqual(EXECUTED, [],
                         "不安全反序列化被执行了——weights_only 回退分支被重新引入")

    def test_rejects_small_structure_mismatch(self):
        """修复前用 strict=False + 「缺失不超过 10%」阈值放行：
        只缺 1 个键时会静默加载，那部分层保持随机初始化，输出掩膜看似正常。

        用官方权重删掉一个键来复现（低于原阈值 max(2, n//10)）。
        """
        import torch

        from src.infer import available_weights
        from src.model_unet import load_checkpoint

        found = available_weights()
        if not found:
            self.skipTest("未找到权重文件")
        payload = torch.load(found[0], map_location="cpu", weights_only=True)
        state = dict(payload["state_dict"])
        state.pop(next(iter(state)))  # 只少一个键
        trimmed = safe_temp_path(self.tmp, "trimmed.pt")
        torch.save({"state_dict": state, "meta": payload.get("meta", {})}, trimmed)

        with self.assertRaises(RuntimeError) as ctx:
            load_checkpoint(trimmed, device="cpu")
        self.assertIn("结构不匹配", str(ctx.exception))

    def test_bundled_weights_load_strictly(self):
        """官方权重必须能通过严格加载，否则收紧校验本身就成了回归。"""
        from src.infer import available_weights
        from src.model_unet import load_checkpoint

        found = available_weights()
        if not found:
            self.skipTest("未找到权重文件")
        model, meta = load_checkpoint(found[0], device="cpu")
        self.assertIsNotNone(model)
        self.assertIn("arch_key", meta)

    def test_synthetic_cache_isolated_per_path(self):
        """修复前用单个布尔缓存，多权重场景下会把真实权重误判成合成。"""
        from src.infer import FloodDetector

        det = FloodDetector(mode="auto")
        self.assertIsInstance(det._synthetic_weights, dict)
        key_a = os.path.normcase(safe_temp_path(self.tmp, "a.pt"))
        key_b = os.path.normcase(safe_temp_path(self.tmp, "b.pt"))
        det._synthetic_weights[key_a] = True
        det._synthetic_weights[key_b] = False
        self.assertNotEqual(det._synthetic_weights[key_a], det._synthetic_weights[key_b])


class TestNpyPickleDisabled(unittest.TestCase):
    def test_read_npy_rejects_object_array(self):
        """修复前 np.load(path) 未显式关闭 pickle，旧版 numpy 会执行对象数组。"""
        target = safe_temp_path(LOGS, "_hardening_obj.npy")
        try:
            buf = io.BytesIO()
            np.save(buf, np.array([1, 2, 3]))
            Path(target).write_bytes(buf.getvalue())
            arr, *_ = preprocess._read_npy(target)
            self.assertEqual(arr.shape[0], 3)

            buf = io.BytesIO()
            np.save(buf, np.array([{"a": 1}], dtype=object))
            Path(target).write_bytes(buf.getvalue())
            with self.assertRaises(Exception):
                preprocess._read_npy(target)
        finally:
            if os.path.exists(target):
                os.remove(target)


# --------------------------------------------------------------------------
# 重投影无数据掩膜（P1 · 假阳性淹没根因）
# --------------------------------------------------------------------------


class TestReprojectNodata(unittest.TestCase):
    def test_reproject_marks_outside_coverage_as_nodata(self):
        """修复前：波段填 0 且 nodata_mask=None，空白区以 NDWI=0 混入识别。

        构造两个错开半个幅面的网格：目标网格右半落在源范围之外，
        这些像元必须被标记为无效，否则会被当成「真实暗像元」参与水体判定。
        """
        src = _mk_scene(x0=0.0, y0=640.0)
        dst = _mk_scene(x0=320.0, y0=640.0)  # 右移 320 m = 32 像元

        out = preprocess.reproject_scene_to(src, dst)

        self.assertIsNotNone(out.nodata_mask, "重投影必须产出无数据掩膜")
        self.assertFalse(out.nodata_mask[:, :16].any(), "源覆盖区内不应判为无数据")
        self.assertGreater(out.nodata_mask[:, 48:].mean(), 0.9, "源范围外应为无数据")
        self.assertGreater(out.nodata_mask.mean(), 0.2, "整体应有可观比例的无数据")
        self.assertIn("reproject_nodata_fraction_pct", out.meta)

    def test_reproject_keeps_nodata_when_source_has_mask(self):
        """源本身带无效区时，重投影后仍应保留（并与覆盖区掩膜合并）。"""
        src = _mk_scene(x0=0.0, y0=640.0)
        src.nodata_mask = np.zeros((64, 64), dtype=bool)
        src.nodata_mask[:32, :] = True  # 上半幅无效
        dst = _mk_scene(x0=0.0, y0=640.0)

        out = preprocess.reproject_scene_to(src, dst)
        self.assertIsNotNone(out.nodata_mask)
        self.assertGreater(out.nodata_mask[:32, :].mean(), 0.9, "源无效区应保持无效")
        self.assertLess(out.nodata_mask[40:, :].mean(), 0.1, "源有效区不应被误判")


# --------------------------------------------------------------------------
# 分块与补边参数校验（P2）
# --------------------------------------------------------------------------


class TestPreprocessGuards(unittest.TestCase):
    def test_iter_tiles_rejects_overlap_not_smaller_than_tile(self):
        """修复前 overlap>=tile 时 stride 退化为 1，切片数按 O(H*W) 爆炸。"""
        arr = np.zeros((64, 64), dtype=np.float32)
        with self.assertRaises(ValueError):
            list(preprocess.iter_tiles(arr, tile=32, overlap=32))
        with self.assertRaises(ValueError):
            list(preprocess.iter_tiles(arr, tile=32, overlap=40))
        with self.assertRaises(ValueError):
            list(preprocess.iter_tiles(arr, tile=0, overlap=0))
        self.assertGreater(len(list(preprocess.iter_tiles(arr, tile=32, overlap=8))), 0)

    def test_pad_to_tile_handles_single_row(self):
        """修复前 reflect 模式在任一维长度为 1 时 numpy 直接报错。"""
        arr = np.zeros((1, 10), dtype=np.float32)
        out, (h, w) = preprocess.pad_to_tile(arr, tile=8)
        self.assertEqual((h, w), (1, 10))
        self.assertEqual(out.shape[1] % 8, 0)

    def test_percentile_stretch_handles_nan(self):
        """修复前 NaN 直接进 astype(uint8)，numpy 会发「invalid value in cast」
        RuntimeWarning 并产生未定义结果。这里把警告升级为异常来捕捉它。"""
        import warnings

        arr = np.array([[0.0, 1.0], [np.nan, np.inf]], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = preprocess.percentile_stretch(arr)
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.isfinite(out.astype(np.float32)).all())

    def test_load_scene_reports_empty_bands_clearly(self):
        """修复前主影像解析不出波段、同时传了近红外文件时，
        `next(iter(bands.values()))` 会先抛 StopIteration，把真正原因
        （波段无法识别）掩盖掉。这里直接打空波段解析结果来触发该分支。
        """
        from unittest import mock

        from PIL import Image

        main_tif = safe_temp_path(LOGS, "_hardening_main.tif")
        nir_tif = safe_temp_path(LOGS, "_hardening_nir.tif")
        try:
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(main_tif)
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(nir_tif)
            with mock.patch.object(preprocess, "resolve_band_order", return_value={}):
                with self.assertRaises(ValueError) as ctx:
                    # 这张小图没有地理参考，rasterio 会发 NotGeoreferencedWarning——
                    # 与本用例无关，屏蔽掉以免干扰输出
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        preprocess.load_scene(main_tif, nir_path=nir_tif)
            self.assertIn("未能从影像解析出任何可用波段", str(ctx.exception))
        finally:
            for target in (main_tif, nir_tif):
                if os.path.exists(target):
                    os.remove(target)


# --------------------------------------------------------------------------
# 路径穿越清洗与输入白名单（P1）
# --------------------------------------------------------------------------


class TestSarPathSafety(unittest.TestCase):
    def test_safe_sid_strips_traversal(self):
        """修复前 sid 未清洗就拼进输出文件名，可写到目标目录之外。

        清洗后应只剩纯文件名片段：不含路径分隔符、不含 ".."，
        因此拼进任何目录都不可能逃出该目录。
        """
        for probe in UNSAFE_NAMES:
            cleaned = sar.safe_sid(probe)
            self.assertNotIn("..", cleaned, f"{probe!r} 未清洗干净")
            self.assertNotIn("/", cleaned)
            self.assertNotIn("\\", cleaned)
        self.assertEqual(sar.safe_sid(UNSAFE_NAMES[0]), "escaped.tif")
        self.assertEqual(sar.safe_sid("", "fallback"), "fallback")
        self.assertEqual(sar.safe_sid(None, "fallback"), "fallback")
        self.assertEqual(sar.safe_sid(""), "sar_sample")  # 未指定时的默认回退名
        self.assertEqual(sar.safe_sid("zhuozhou2023"), "zhuozhou2023")
        self.assertLessEqual(len(sar.safe_sid("a" * 500)), 64)

    def test_polarization_whitelist(self):
        """修复前极化方式直接进 glob 与文件名，可匹配非预期文件或逃出目录。"""
        self.assertEqual(sar.normalize_polarization("vv"), "VV")
        self.assertEqual(sar.normalize_polarization("VH"), "VH")
        for bad in ("V*", "../x", "XX", "", "vv;rm", "*"):
            with self.assertRaises(ValueError, msg=f"{bad!r} 应被拒绝"):
                sar.normalize_polarization(bad)


# --------------------------------------------------------------------------
# 界面层容错（P2）
# --------------------------------------------------------------------------


class TestComponentsHardening(unittest.TestCase):
    def test_load_manifest_skips_corrupt_json(self):
        """修复前 json.load 无捕获，而 main.py 模块级就调用它——
        一份半写入的 samples.json 会让应用直接起不来。"""
        from app import components

        with tempfile.TemporaryDirectory() as d:
            manifest = safe_temp_path(d, "samples.json")
            _write_text(manifest, "{ this is not valid json")
            self.assertEqual(components.load_manifest([d]), [])

    def test_load_manifest_filters_bad_entries(self):
        from app import components

        with tempfile.TemporaryDirectory() as d:
            payload = {"samples": [
                {"id": "good", "pre": "a.tif", "post": "b.tif"},
                {"no_id": 1},
                "junk",
            ]}
            manifest = safe_temp_path(d, "samples.json")
            _write_text(manifest, json.dumps(payload, ensure_ascii=False))
            out = components.load_manifest([d])
            self.assertEqual([s["id"] for s in out], ["good"])

    def test_stats_html_tolerates_incomplete_change(self):
        """修复前直接下标 c["before_water_km2"]，缺字段抛 KeyError——
        此时成果包已导出，界面却报失败。"""
        from app import components

        html = components.stats_html(_fake_result(change={"new_water_km2": 0.5}))
        self.assertIn("heye-stats", html)
        self.assertIn("新增淹没面积", html)

    def test_stats_html_tolerates_none_values(self):
        from app import components

        res = _fake_result(
            stats={"water_area_km2": None, "water_fraction_pct": "x"},
            change={"before_water_km2": None, "after_water_km2": 10.0,
                    "new_water_km2": None, "receded_water_km2": None},
        )
        self.assertIn("km²", components.stats_html(res))

    def test_num_and_esc_helpers(self):
        from app import components

        self.assertEqual(components._num(None), 0.0)
        self.assertEqual(components._num("bad"), 0.0)
        self.assertEqual(components._num(float("nan")), 0.0)
        self.assertEqual(components._num("2.5"), 2.5)
        self.assertNotIn("<script>", components._esc("<script>alert(1)</script>"))
        self.assertIn("&lt;script&gt;", components._esc("<script>alert(1)</script>"))

    def test_export_result_falls_back_to_temp_dir(self):
        """修复前默认写 ROOT/outputs（打包后的只读资源目录），
        装到 Program Files 时导出必然 OSError 且界面只回一句「失败」。"""
        from app import components
        from src import report as report_mod

        seen = []
        real = report_mod.export_bundle

        def fake_bundle(result, out_dir="outputs", **kw):
            seen.append(out_dir)
            if len(seen) == 1:
                raise OSError("read-only filesystem")
            return {"zip": out_dir + "x.zip", "files": {},
                    "workdir": out_dir, "report": ""}

        report_mod.export_bundle = fake_bundle
        try:
            zip_path = components.export_result(object(), out_dir="Z:/readonly/")
        finally:
            report_mod.export_bundle = real

        self.assertEqual(len(seen), 2, "首次失败后应重试到备用目录")
        self.assertNotEqual(seen[0], seen[1])
        self.assertIn("huiyan_", seen[1])
        self.assertIn("huiyan_", zip_path)


# --------------------------------------------------------------------------
# 界面注入与参数校验（P2）
# --------------------------------------------------------------------------


class TestMainHardening(unittest.TestCase):
    def test_html_escape_helpers(self):
        from app import main as main_mod

        self.assertIn("&lt;b&gt;", main_mod._esc("<b>"))
        self.assertEqual(main_mod._safe_float(None), 0.0)
        self.assertEqual(main_mod._safe_float("1.5"), 1.5)
        self.assertEqual(main_mod._safe_float(float("inf")), 0.0)

    def test_sar_stats_html_escapes_remote_metadata(self):
        """修复前 pre_scene / aoi_source 直接来自 STAC 元数据就拼进 HTML。"""
        from app import main as main_mod

        result = {
            "stats": {"new_water_km2": 1.0, "window_km2": 100.0, "threshold_db": -16.0,
                      "pre_water_km2": 1.0, "post_water_km2": 2.0,
                      "persistent_km2": 1.0, "receded_km2": 0.1},
            "provenance": {
                "aoi_lonlat": [115.9, 39.4], "polarization": "VV",
                "window": [512, 512], "aoi_source": "<img onerror=alert(1)>",
                "pre_scene": "<script>evil()</script>", "post_scene": "b",
            },
        }
        html = main_mod._sar_stats_html(result)
        # 尖括号必须被转义：转义后即使保留 onerror=alert 字面量也不再是标签属性
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&lt;img", html)

    def test_sar_stats_html_tolerates_missing_fields(self):
        from app import main as main_mod

        # 缺字段时不应抛 KeyError（原先大量直接下标）
        self.assertIn("heye-stats", main_mod._sar_stats_html({"stats": {}, "provenance": {}}))

    def test_coerce_pixel_size_rejects_bad_input(self):
        """修复前非法输入静默返回 None（改用影像分辨率），用户得不到任何提示。"""
        import gradio as gr

        from app import main as main_mod

        self.assertIsNone(main_mod._coerce_pixel_size(""))
        self.assertIsNone(main_mod._coerce_pixel_size(0))
        self.assertEqual(main_mod._coerce_pixel_size(10), 10.0)
        for bad in ("abc", -5, float("nan")):
            with self.assertRaises(gr.Error):
                main_mod._coerce_pixel_size(bad)

    def test_resolve_weights_reports_missing_file(self):
        """修复前找不到权重时静默返回 None——界面显示 U-Net、实际跑基线。"""
        import gradio as gr

        from app import main as main_mod

        self.assertIsNone(main_mod._resolve_weights(""))
        self.assertIsNone(main_mod._resolve_weights("（自动）"))
        with self.assertRaises(gr.Error):
            main_mod._resolve_weights("definitely-not-a-real-weight-file.pt")


# --------------------------------------------------------------------------
# 数据抓取加固（P1/P2）
# --------------------------------------------------------------------------


class TestFetchHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "frs_under_test", os.path.join(ROOT, "scripts", "fetch_real_samples.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls.frs = mod

    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="huiyan_manifest_"))

    def test_scl_cloud_and_water_use_valid_pixels(self):
        """半幅无数据时，云/水比例不能被空白像元稀释。"""
        scl = np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [9, 9, 6, 6],
                        [4, 4, 4, 4]], dtype=np.uint8)
        quality = self.frs._scl_quality(scl)
        self.assertEqual(quality["valid_pct"], 50.0)
        self.assertEqual(quality["cloud_pct"], 25.0)
        self.assertEqual(quality["water_pct"], 25.0)

        empty = self.frs._scl_quality(np.zeros((2, 2), dtype=np.uint8))
        self.assertEqual(empty["valid_pct"], 0.0)
        self.assertEqual(empty["cloud_pct"], 100.0)

    def test_private_ip_detection(self):
        """SSRF 防护：解析出的地址落在内网/环回/链路本地时必须拦截。"""
        for ip in ("127.0.0.1", "10.0.0.1", "192.168.1.1", "172.16.0.1",
                   "169.254.169.254", "0.0.0.0", "::1", "not-an-ip"):
            self.assertTrue(self.frs._is_private_ip(ip), f"{ip} 应判为不可信")
        self.assertFalse(self.frs._is_private_ip("8.8.8.8"))
        self.assertFalse(self.frs._is_private_ip("1.1.1.1"))

    def test_asset_url_scheme_and_host_whitelist(self):
        """修复前 href 来自 STAC 响应就直接交给 urlopen/GDAL。"""
        check = self.frs._assert_public_https_url
        for bad in ("http://evil.com/x.tif", "file:///etc/passwd",
                    "https://evil.internal/x.tif", "https://127.0.0.1/x.tif",
                    "https://10.0.0.1/x.tif"):
            with self.assertRaises(ValueError, msg=f"{bad!r} 应被拒绝"):
                check(bad)

    def test_planetary_computer_blob_hosts_are_allowed(self):
        """回归用例：PC 的 COG 资产放在 Azure Blob 上，主机名带账号前缀
        且会变（sentinel2l2a01 等）。曾用精确匹配白名单，结果把整个 PC
        数据源拦死——实测在选景阶段把所有 PC 候选景逐个跳过，
        只能退回 AWS 兜底。白名单必须按域后缀放行。
        """
        allowed = self.frs._host_allowed
        for host in ("sentinel2l2a01.blob.core.windows.net",
                     "sentinel1euwestrtc01.blob.core.windows.net",
                     "ai4edataeuwest.blob.core.windows.net",
                     "sentinel-s2-l2a.s3.amazonaws.com",
                     "sentinel-cogs.s3.us-west-2.amazonaws.com",
                     "earth-search.aws.element84.com",
                     "planetarycomputer.microsoft.com"):
            self.assertTrue(allowed(host), f"{host} 应被放行")

        for host in ("evil.internal", "", "127.0.0.1",
                     "blob.core.windows.net.evil.com",
                     "evil-blob.core.windows.net",   # 前缀不是点，不能算同一域
                     "notblob.core.windows.net.attacker.net"):
            self.assertFalse(allowed(host), f"{host} 应被拒绝")

    def test_rejected_asset_is_not_retried(self):
        """白名单拒绝属确定性失败，重试不会改变结果——
        原实现会让每个被拒的候选景白等 2+4 秒并刷日志。"""
        frs = self.frs
        self.assertTrue(issubclass(frs.AssetRejected, ValueError),
                        "应仍是 ValueError 子类，调用方既有捕获逻辑不受影响")
        calls = []

        def rejected():
            calls.append(1)
            raise frs.AssetRejected("拒绝白名单之外的资产主机")

        with self.assertRaises(frs.AssetRejected):
            frs._with_retry(rejected, attempts=3, base_delay=0.01, label="t")
        self.assertEqual(len(calls), 1, "确定性失败不应重试")

    def test_with_retry_propagates_timeout_without_retrying(self):
        """修复前 except Exception 把 deadline 的 TimeoutError 也当网络抖动重试。"""
        calls = []

        def boom():
            calls.append(1)
            raise TimeoutError("deadline reached")

        with self.assertRaises(TimeoutError):
            self.frs._with_retry(boom, attempts=3, base_delay=0.01, label="t")
        self.assertEqual(len(calls), 1, "超时不应被重试")

    def test_with_retry_retries_other_errors(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("transient")
            return "ok"

        self.assertEqual(self.frs._with_retry(flaky, attempts=3, base_delay=0.01), "ok")
        self.assertEqual(len(calls), 3)

    def test_sign_cache_expires(self):
        """修复前缓存永不过期，SAS token 失效后持续 403 且无法自愈。"""
        frs = self.frs
        frs._SIGN_CACHE.clear()
        calls = []

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                calls.append(1)
                return json.dumps(
                    {"href": "https://planetarycomputer.microsoft.com/asset?sig=abc"}
                ).encode()

        real_urlopen = frs.urllib.request.urlopen
        real_assert = frs._assert_public_https_url
        frs.urllib.request.urlopen = lambda *a, **k: _Resp()
        frs._assert_public_https_url = lambda u: u  # 本用例只验证缓存行为
        try:
            href = "https://sentinel-s2-l2a.s3.amazonaws.com/tiles/t.tif"
            first = frs._sign_pc(href)
            second = frs._sign_pc(href)
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1, "第二次应命中缓存")

            # 手动令缓存过期：应重新签名，而不是永远返回失效 token
            value, _expiry = frs._SIGN_CACHE[href]
            frs._SIGN_CACHE[href] = (value, 0.0)
            frs._sign_pc(href)
            self.assertEqual(len(calls), 2, "缓存过期后应重新签名")
        finally:
            frs.urllib.request.urlopen = real_urlopen
            frs._assert_public_https_url = real_assert
            frs._SIGN_CACHE.clear()

    def test_pick_scene_stac_skips_dirty_items(self):
        """修复前 item["bbox"] / item["properties"]["datetime"] 直接下标，
        STAC 条目缺字段会 KeyError 中断整轮候选遍历。"""
        items = [
            {"id": "no_bbox", "properties": {"datetime": "2023-08-05T00:00:00Z"}},
            {"id": "no_datetime", "bbox": [115.0, 39.0, 116.0, 40.0], "properties": {}},
            {"id": "bad_date", "bbox": [115.0, 39.0, 116.0, 40.0],
             "properties": {"datetime": "not-a-date"}},
            {"id": "null_cloud", "bbox": [115.0, 39.0, 116.0, 40.0],
             "properties": {"datetime": "2023-08-05T00:00:00Z", "eo:cloud_cover": None}},
        ]
        # 这些条目都不会进入联网环节，应被安静淘汰（返回 None）而不是抛异常
        self.assertIsNone(self.frs._pick_scene_stac(items, 115.97, 39.49, "2023-08-05", 60.0))

    def test_write_json_atomic_produces_valid_json_without_tmp(self):
        """清单原子写：修复前直接覆写，中途崩溃会留下截断 JSON，
        下次加载解析失败只能按空清单处理，等于整批样本记录丢失。

        原子写法应保证：目标文件始终是完整 JSON，且不残留 .tmp。
        """
        frs = self.frs
        target = safe_temp_path(self.tmp, "samples.json")
        _write_text(target, "{ 这是上次崩溃留下的半截文件")

        payload = {"samples": [{"id": "a"}]}
        frs.write_json_atomic(target, payload)
        self.assertEqual(json.loads(Path(target).read_text(encoding="utf-8")), payload)
        self.assertEqual([n for n in os.listdir(self.tmp) if n.endswith(".tmp")], [],
                         "原子写不应残留 .tmp 文件")

        # 覆盖已有完整文件同样应成功
        frs.write_json_atomic(target, {"samples": [{"id": "b"}]})
        again = json.loads(Path(target).read_text(encoding="utf-8"))
        self.assertEqual(again["samples"][0]["id"], "b")


class TestS1RtcPadding(unittest.TestCase):
    """Sentinel-1 RTC 补边掩膜（P1）。

    窗口贴影像边界时 `read_rtc_window` 会用 0 补齐到请求尺寸。该数组是**线性 σ0**，
    0 换算成 dB 约 -60 dB，远低于水体阈值，若不随数组返回"哪些是补边"的掩膜，
    下游会把这些像元判成水，窗口边缘出现大片假淹没、面积统计虚高。
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "s1rtc_under_test", os.path.join(ROOT, "scripts", "fetch_s1_rtc.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls.rtc = mod

    def test_padded_area_is_marked_invalid(self):
        from unittest import mock

        from rasterio.transform import from_origin

        affine = from_origin(0.0, 1000.0, 10.0, 10.0)

        class _FakeSrc:
            crs = "EPSG:32650"
            # 注意：类体内 `transform = transform` 的右侧会解析到类命名空间自身，
            # 必须换一个名字引用外层变量
            transform = affine

            def read(self, idx, window=None):
                # 只读回 90×80，小于请求的 100×100 —— 触发补边分支
                return np.full((90, 80), 0.5, dtype=np.float32)

            def window_transform(self, win):
                return affine

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        with mock.patch("rasterio.open", return_value=_FakeSrc()), \
                mock.patch.object(self.rtc, "log", lambda *a, **k: None):
            arr, _tf, _crs, valid = self.rtc.read_rtc_window(
                "/fake/scene.tif", "token", 116.0, 39.0, 100
            )

        self.assertEqual(arr.shape, (100, 100), "补边后应达到请求尺寸")
        self.assertEqual(int(valid.sum()), 90 * 80, "只有真实读到的像元算有效")
        self.assertTrue(bool(valid[:90, :80].all()), "真实区域应有效")
        self.assertFalse(bool(valid[90:, :].any()), "补边行必须标记为无效")
        self.assertFalse(bool(valid[:, 80:].any()), "补边列必须标记为无效")


class TestTrainChannelOrder(unittest.TestCase):
    """训练数据集通道顺序（P1）。

    修复前先过滤缺失波段、再把补零通道追加到末尾：缺 red 时实际堆叠成
    [blue, green, nir, 0]，而模型与归一化仍按 [blue, green, red, nir] 解释——
    训练不报错，只是精度莫名下降；推理端复用同一逻辑会直接给出错误掩膜。
    """

    @classmethod
    def setUpClass(cls):
        import train as train_mod

        cls.train = train_mod

    def setUp(self):
        from PIL import Image

        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="huiyan_dataset_"))
        mask = safe_temp_path(self.tmp, "mask.png")
        Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(mask)
        self.mask = mask

    def _four_band_npy(self) -> str:
        target = safe_temp_path(self.tmp, "four_band.npy")
        buf = io.BytesIO()
        np.save(buf, np.zeros((32, 32, 4), dtype=np.float32))
        Path(target).write_bytes(buf.getvalue())
        return target

    def test_four_band_image_is_loaded(self):
        ds = self.train.FloodDataset(
            [{"image": self._four_band_npy(), "mask": self.mask}], img_size=8, augment=False
        )
        chw, mask = ds._load(0)
        self.assertEqual(chw.shape[0], 4, "4 波段影像应堆叠成 4 通道")
        self.assertEqual(tuple(mask.shape), (32, 32))

    def test_missing_band_is_rejected_with_clear_message(self):
        """3 波段（缺近红外）必须直接报错，指出缺哪个波段，而不是补零糊过去。"""
        from PIL import Image

        target = safe_temp_path(self.tmp, "three_band.tif")
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(target)
        ds = self.train.FloodDataset(
            [{"image": target, "mask": self.mask}], img_size=8, augment=False
        )
        with self.assertRaises(ValueError) as ctx:
            # 这张小图没有地理参考，rasterio 会发 NotGeoreferencedWarning——
            # 与本用例无关，屏蔽掉以免干扰输出
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds._load(0)
        self.assertIn("nir", str(ctx.exception))


# --------------------------------------------------------------------------
# 命中本地缓存的分析路径（点地图预设点走的就是这条路）
# --------------------------------------------------------------------------


class TestPipelineCachePath(unittest.TestCase):
    """本地缓存分支的端到端回归。

    回归背景：给结果补卫星影像溯源时，代码引用了 `entry`——而它只在
    "联网抓取"分支里赋值。命中本地缓存时该分支被跳过，于是抛
    `UnboundLocalError`，把演示路径（点预设点秒出结果）整个打坏。
    当时 130 项用例全过却没抓到，因为没有任何用例走过这个分支。
    """

    def setUp(self):
        from app.main import _local_optical_pair

        if _local_optical_pair("poyang2020") is None:
            self.skipTest("缺少 poyang2020 本地缓存影像，跳过")

    def test_cached_event_pipeline_runs_end_to_end(self):
        from app.main import _pipeline_body

        logs: list = []
        out = _pipeline_body(
            116.30, 29.15,
            "2020-05-08", "2020-06-05",
            "2020-07-10", "2020-07-30",
            512, logs.append,
        )
        self.assertEqual(len(out), 5, "应返回 5 个界面输出项")
        text = "\n".join(logs)
        self.assertIn("跳过下载", text, "预设点应命中本地缓存")
        self.assertIn("成果包", text)
        self.assertIn("km²", str(out[3]), "结论文本应含面积统计")

        zip_path = str(out[4])
        self.assertTrue(os.path.isfile(zip_path), f"成果包不存在：{zip_path}")
        self.assertGreater(os.path.getsize(zip_path), 0, "成果包不应为空")


# --------------------------------------------------------------------------
# 端口探测
# --------------------------------------------------------------------------


class TestDesktopPort(unittest.TestCase):
    def test_find_free_port_returns_bindable_port(self):
        from app.desktop import find_free_port

        port = find_free_port("127.0.0.1", 47100, tries=30)
        self.assertGreaterEqual(port, 47100)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))  # 应能成功绑定


if __name__ == "__main__":
    unittest.main(verbosity=2)
