你正在「慧眼识灾」仓库 F:\deepseek\flood-eyes 中工作。请阅读 GEMINI_REVIEW.md，然后动手完成审核与优化，不要只给意见。

硬性约束：
- 默认模型保持 NDWI+Otsu 基线；不要把默认改成合成数据训练的 U-Net。
- 不要删除诚实声明。
- 不要重训大模型、不要下载 Sen1Floods11 全集。
- 改完必须运行：python tests/test_pipeline.py
- 只改本仓库源码与测试，不要动 D:\慧眼识灾 安装目录。

优先检查并修复：
1. 面积、配准、BOA 偏移、云掩膜、波段顺序、SAR 标定 XML / UTM、权重 arch 加载是否还会静默失败。
2. 一键全流程（下载→识别）易失败点、日志是否及时、默认事件是否合理。
3. UI（app/apple.css、app/main.py）明显的 Gradio 覆盖问题。
4. 补上能写的回归测试。

完成后用中文写一份简短总结：改了哪些文件、测没测过、还有哪些没动。
