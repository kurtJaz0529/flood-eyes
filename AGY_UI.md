你只改「慧眼识灾」前端外观，仓库 F:\deepseek\flood-eyes。目标：高级、简约，接近 Apple / Linear 质感，不要花哨渐变、不要科技蓝紫、不要 emoji 堆砌。

可改文件（仅这些）：
- app/apple.css
- app/components.py（统计卡 HTML/CSS）
- app/main.py 里与 HEADER、FOOTER、_SAR_CSS、文案、theme、控件 label 相关的部分
不要改识别算法、下载逻辑、默认事件（保持 poyang2020）、不要改 NDWI 为默认模型。

设计要求：
1. 浅色：背景 #F5F5F7，卡片纯白，文字 #1D1D1F，次要 #86868B，唯一强调色系统蓝 #007AFF。
2. 大留白、细分割线、大圆角（16–20px）、主按钮胶囊形。
3. 字体 -apple-system / PingFang SC / Microsoft YaHei UI；数字字重 600、字距略收。
4. CSS 选择器必须收窄到 .gradio-container，禁止 `button:not(.primary)` 这类全局选择器，以免打坏 Gradio 工具栏。
5. 页签做成分段控件（segmented control）。
6. 地图 iframe、图片圆角与卡片一致。
7. 统计面板（stats_html / SAR 卡）去掉厚重渐变，白底大数字。
8. 文案缩短：按钮「开始分析」等保持克制。

改完用中文说明改了哪些视觉点。不要跑下载，不要重训。
