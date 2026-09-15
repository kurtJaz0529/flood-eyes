# 大赛执行清单 · 慧眼识灾

> 对照原方案 M0–M4 里程碑，逐项勾选。**已完成项都是仓库里可验证的代码/文件**。

---

## M0 环境（半天）✅

| 事项 | 状态 | 验证方式 |
|---|---|---|
| 检查 Python / GPU | ✅ | `python --version`、`python -c "import torch; print(torch.cuda.is_available())"` |
| 虚拟环境 + 依赖安装 | ✅ | `pip install -r requirements.txt` |
| 环境自检脚本 | ✅ | `python tests/test_pipeline.py`（35 项用例） |
| 本机实测 | ✅ | Python 3.13.7 + torch 2.12.1(CPU) 全流程跑通 |

**注意**：本机 torch 是 CPU 版。训练建议换 CUDA 版：
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## M1 基线（1 天）✅

| 事项 | 状态 | 文件 |
|---|---|---|
| 项目脚手架 | ✅ | `src/`、`app/`、`train.py` |
| 波段解析 + NDWI | ✅ | `src/preprocess.py` |
| NDWI + Otsu 基线 | ✅ | `src/baseline.py`（含 NIR 闸门、退化场景保护） |
| 后处理 + 面积统计 | ✅ | `src/postprocess.py` |
| 统一推理接口 | ✅ | `src/infer.py` |
| Gradio 初版 | ✅ | `app/main.py` + `app/components.py` |
| 演示样本 | ✅ | `data/make_samples.py` → `data/samples/` |
| **真实影像抓取** | ✅ | `scripts/fetch_real_samples.py` → `data/real/`（鄱阳湖 2020、涿州 2023，含溯源） |
| **产出：可演示 v0.1** | ✅ | `python app/main.py` |

---

## M2 模型（2 天）✅

| 事项 | 状态 | 文件 |
|---|---|---|
| U-Net 模型（smp + 纯 torch 兜底） | ✅ | `src/model_unet.py` |
| 训练脚本（Dice+BCE、场景级划分、AMP） | ✅ | `train.py` |
| IoU/F1 曲线记录 | ✅ | `logs/train_log.csv`、`logs/train_curve.png` |
| 权重存取（含归一化统计量） | ✅ | `weights/best_model.pt` |
| 合成数据跑通 | ✅ | 20 epoch / 46 秒 / val IoU 0.971 |
| **待办：Sen1Floods11 真实训练** | ⬜ | `python scripts/download_data.py --dataset sen1floods11` |

**真实训练建议参数**（单卡 8 GB）：
```powershell
python train.py --data data/sen1floods11 --epochs 60 --img-size 512 `
    --arch smp --encoder resnet34 --encoder-weights imagenet `
    --batch-size 8 --lr 3e-4 --patience 15
```

---

## 真实数据工程（额外收获，答辩加分项）✅

| 事项 | 状态 | 说明 |
|---|---|---|
| 免注册抓取真实 Sentinel-2 | ✅ | `scripts/fetch_real_samples.py`（STAC 检索 + COG 窗口读，只下 10 km 窗口） |
| 真实灾前/灾后样本 | ✅ | `data/real/`：鄱阳湖 2020（44.02 → 154.37 km²）、涿州 2023（1.33 → 5.26 km²） |
| BOA 偏移量判定 | ✅ | 与原始 JP2 逐窗比对；实测 COG 中位 DN 589 / JP2 1589，差 −1000 |
| 旋转瓦片 nodata 规避 | ✅ | 用 SCL 缩略图定位最近有效像元，自动平移窗口 |
| 云量拒绝机制 | ✅ | AOI 窗口云量超阈值不出图（洞庭湖 2024 因此被拒，最低 33.5%） |
| 真实影像回归测试 | ✅ | `tests/test_pipeline.py` 新增 4 项（溯源完整性、反射率物理性、几何一致性、对比可运行） |

**答辩素材**：这三个坑（偏移量、旋转瓦片、汛期云遮挡）是真实工程问题，
比"调了个模型"更能体现团队的数据素养 —— 尤其**偏移量判定**，做错了整景 NDWI 直接失效。

---

## M3 界面（1 天）✅

| 事项 | 状态 | 说明 |
|---|---|---|
| 双时相对比 | ✅ | 页签②，新增淹没/退水/持续水体 |
| 面积报表 | ✅ | 统计卡片 + `stats.json` |
| PDF 简报导出 | ✅ | `src/report.py`（中文 CID 字体，无需外部字体） |
| 对比滑块 | ✅ | Gradio `ImageSlider`，老版本自动降级 |
| 成果包 zip 一键下载 | ✅ | 掩膜/叠加图/热力图/JSON/PDF |
| 界面美化 | ✅ | 自定义卡片 CSS、渐变主指标 |
| 演示 GIF | ✅ | `scripts/make_demo_gif.py` → `docs/assets/demo_real_poyang.gif` |
| **桌面应用（exe）** | ✅ | `build/build_app.ps1`；精简版 427 MB / 完整版 878 MB；`scripts/verify_packaged_app.py` 10/10 通过 |
| **标准安装包（setup.exe）** | ✅ | `build/build_installer.ps1`；精简版 174.6 MB / 完整版 351.9 MB；安装→运行→卸载全流程实测 |
| **产出：路演级 demo** | ✅ | — |

---

## M4 材料（半天）✅

| 事项 | 状态 | 文件 |
|---|---|---|
| GitHub 仓库结构 | ✅ | 见 README「仓库结构」 |
| README（门面） | ✅ | `README.md` |
| 演示 GIF | ✅ | `docs/assets/demo_real_poyang.gif` |
| 3 分钟路演脚本 | ✅ | `docs/demo_script.md` |
| 答辩 PPT 大纲 | ✅ | `docs/pitch_outline.md` |
| 本执行清单 | ✅ | `docs/competition_checklist.md` |
| **待办：录屏 3 分钟** | ⬜ | 按 `docs/demo_script.md` 逐句录 |
| **待办：仓库推 GitHub** | ⬜ | 见下方推送清单 |

---

## 仓库推送前检查

- [ ] `weights/*.pt` 已被 `.gitignore` 排除（大文件走 Release 或网盘）
- [ ] `data/sen1floods11/`、`outputs/`、`logs/*.log` 未入库
- [ ] 权重若必须入库，用 Git LFS
- [ ] README 里的 GIF 路径正确（`docs/assets/demo_real_poyang.gif`）
- [ ] 跑一遍 `python tests/test_pipeline.py` 全绿
- [ ] 删除本机绝对路径、临时脚本（`_t*.py`）
- [ ] 补充 LICENSE（建议 MIT 或 Apache-2.0）
- [ ] 补团队信息、指导教师、学校

```powershell
git init
git add .
git commit -m "feat: 慧眼识灾 v0.1 —— NDWI+Otsu 基线 + U-Net + Gradio 演示 + PDF 简报"
git branch -M main
git remote add origin <你的仓库地址>
git push -u origin main
```

---

## 风险预案（现场演示）

| 风险 | 预案 |
|---|---|
| 网络断开 | 全部依赖本地文件，示例样本已入库；不联网也能演示 |
| 没有 GPU | 基线模型 CPU 秒级；U-Net CPU 单景 <1 秒 |
| 权重文件损坏 | 系统自动回落基线并提示，不会白屏 |
| 评委要求现场换数据 | 支持直接上传 `.tif/.png/.npy`，自动识别波段 |
| 界面启动失败 | 录屏备份 + `python tests/test_pipeline.py` 证明核心可用 |

---

## 评审维度对照

| 评审关注 | 我们的证据 |
|---|---|
| 创新性 | 近红外物理闸门、双模型策略、双时相变化检测 |
| 可行性 | 全流程可运行代码 + 35 项自检 + 零采购成本数据 |
| 社会价值 | 救援时效、基层可及性、公共数据价值释放 |
| 完成度 | M0–M3 全部落地，M4 材料齐备 |
| 真实性 | 合成数据指标明确标注，不冒充真实精度 |
