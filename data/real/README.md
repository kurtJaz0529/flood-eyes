# 真实卫星影像样本 · 数据溯源

⚠️ 与 `data/samples/`（合成数据）不同，**本目录全部是真实 Sentinel-2 影像**，
用于真实场景演示与算法验证。**没有人工标注掩膜**，因此不用于计算 IoU。

## 数据来源

| 项目 | 说明 |
|---|---|
| 卫星/产品 | Sentinel-2 L2A（大气校正后地表反射率，10 m） |
| 数据托管 | AWS 公开桶 `sentinel-cogs`（Cloud-Optimized GeoTIFF，免注册） |
| 检索接口 | Element84 Earth Search STAC：`https://earth-search.aws.element84.com/v1` |
| 许可 | Copernicus Sentinel Data Terms and Conditions（免费、可商用） |
| 获取脚本 | `python scripts/fetch_real_samples.py --event all` |

## 本目录的样本

| 样本 | 事件 | 灾前景 | 灾后景 | AOI 窗口云量 |
|---|---|---|---|---|
| `zhuozhou2023` | 2023·河北涿州暴雨洪涝 | `S2A_50SMJ_20230716_0_L2A`（2023-07-16） | `S2A_50SMJ_20230815_0_L2A`（2023-08-15） | 0.0% / 0.0% |
| `poyang2020` | 2020·江西鄱阳湖特大洪水 | `S2A_50RMT_20200519_1_L2A`（2020-05-19） | `S2A_50RMT_20200715_0_L2A`（2020-07-15） | 4.3% / 0.1% |

- 窗口尺寸：1280×1280 像元 = 12.8 km × 12.8 km = 163.84 km²
- 坐标系：随景（EPSG:32650 / UTM 50N）
- 云量：由 SCL 场景分类图在 AOI 窗口内统计（云+云影类别 3/8/9/10）

## 文件说明

| 文件 | 内容 |
|---|---|
| `{id}_pre.tif` / `{id}_post.tif` | 4 波段（B2 蓝 / B3 绿 / B4 红 / B8 近红外），uint16，**反射率 ×10000** |
| `{id}_pre_scl.png` / `{id}_post_scl.png` | SCL 场景分类图（水体蓝、云白、云影灰、植被绿） |
| `{id}_preview.jpg` | 灾前/灾后真彩预览（联合百分位拉伸） |
| `samples.json` | 景号、时间、云量、AOI、坐标系、偏移量处理方式等完整溯源信息 |

## ⚠️ 两个必须知道的坑（脚本已处理）

### 1. BOA 偏移量（baseline ≥ 04.00 的 −1000）

2022-01-25 之后 ESA 的 L2A 产品带 −1000 的 BOA 偏移。Earth Search 的 COG **有的已扣、有的没扣**，
而 STAC 里的 `earthsearch:boa_offset_applied` 标记**存在已知错误**
（[Element84 issue #66](https://github.com/Element84/earth-search/issues/66)、[#71](https://github.com/Element84/earth-search/issues/71)）。

处理错误会让整景影像系统性偏移 0.1 反射率，**NDWI 直接失效**。

本脚本的做法：**与原始 JP2 逐窗比对**——同一窗口若
`中位DN(COG) − 中位DN(JP2) ≈ −1000`，则判定 COG 已扣偏移，反射率直接取 `DN/10000`。
本目录两景的判定结果记录在 `samples.json` 的 `*_offset_basis` 字段。

### 2. UTM 瓦片是旋转的

瓦片包围盒的角上是 nodata。若 AOI 落在角上，按经纬度直接取窗口会读出一片 0。
脚本用 SCL 缩略图寻找最近的**有效像元**，把窗口中心平移过去（偏移量记在 `*_window_shift_km`）。

## 识别结果（NDWI + Otsu 基线 + 近红外闸门）

| 样本 | 灾前水体 | 灾后水体 | 新增淹没 |
|---|---|---|---|
| 涿州 2023 | 1.33 km² | 5.26 km² | **3.98 km²** |
| 鄱阳湖 2020 | 44.02 km² | 154.37 km² | **110.72 km²** |

> 这两组数字可与公开灾情报道交叉验证（涿州城区淹没、鄱阳湖水域面积较汛前扩大数倍）。
> 但**这不是精度评估**——没有逐像元真值，只有"量级合理"的定性验证。

## 缺失的事件：2024·洞庭湖团洲垸决口

脚本尝试抓取但**失败了**，原因值得写进答辩材料：

> 2024-07-06 ~ 07-25 期间，AOI 窗口云量最低的一景仍有 **33.5% 被云遮挡**
> （最差 99.9%），超过 20% 阈值，脚本拒绝出图。

这正是光学遥感的固有短板——**汛期云雨同步，洪水来了、光学卫星也看不见**。
下一步必须接入 Sentinel-1 雷达（SAR 穿云），这也是 README Roadmap 里的第一项。

复现：
```powershell
python scripts/fetch_real_samples.py --event dongting2024 --max-cloud 40 --allow-cloudy
```

## 想抓更多事件？

编辑 `scripts/fetch_real_samples.py` 顶部的 `EVENTS` 字典，加一条：

```python
"my_event": {
    "label": "2025·某地洪水",
    "aoi": (经度, 纬度),
    "pre":  ("起", "止", "目标日期"),
    "post": ("起", "止", "目标日期"),
    "size": 1280,
    "note": "灾情简述",
}
```

然后 `python scripts/fetch_real_samples.py --event my_event`。
脚本会自动：检索全部过境景 → 逐景评估 AOI 窗口云量 → 挑最优景 → 判定 BOA 偏移 → 落盘 4 波段 GeoTIFF。
