# 请 Gemini 审核并优化「慧眼识灾」

你是资深遥感 + Python 应用审稿人。请阅读本仓库并给出**可落地的修改**，不要空泛夸奖。

## 项目是什么

本地洪水识别软件（Gradio 桌面端）：Sentinel-2 光学 / Sentinel-1 雷达 → 水体掩膜 → 面积 km² → PDF/zip。
默认模型是 NDWI+Otsu 基线（本机、非大模型）。U-Net 仅合成数据训练，不能当竞赛精度。

根目录：`F:\deepseek\flood-eyes`

优先读：

- `src/infer.py` `src/preprocess.py` `src/sar.py` `src/baseline.py` `src/postprocess.py` `src/model_unet.py`
- `app/main.py` `app/components.py` `app/apple.css` `app/desktop.py`
- `scripts/fetch_real_samples.py` `scripts/fetch_s1_rtc.py`
- `tests/test_pipeline.py`

## 请你做的事

1. **正确性**：面积、配准、BOA 偏移、云、波段顺序、SAR 标定/UTM、权重 `arch` 加载是否还会静默出错。
2. **产品**：一键全流程（下载→识别）是否易失败、日志是否及时、默认事件是否合理。
3. **UI**：`app/apple.css` 是否真像高质量桌面软件；Gradio 覆盖是否fragile。
4. **直接改代码**：高优先级 bug 直接修；风格问题列清单。改完补回归测试。
5. **不要**：重训大模型、不要把默认改成合成 U-Net、不要删诚实声明。

## 已知背景（供对照，请独立验证）

已修过：smp_unet 权重映射、地理坐标像元尺寸、SAR 配准与标定 XML、BOA 启发式、云量提示、双时相重投影、默认基线、小孔填充、全流程线程刷日志、本机缓存跳过下载、OSM 地图选点。

## 输出格式

- 先给 5 条以内「必须改」
- 再给「建议改」
- 然后动手改必须项
- 最后说明怎么跑 `python tests/test_pipeline.py` 验证
