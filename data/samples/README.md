# 演示样本说明（合成数据）

> 📌 **需要真实卫星影像？** 见 [`../real/README.md`](../real/README.md) ——
> 仓库已包含两组真实 Sentinel-2 灾前/灾后影像（鄱阳湖 2020、涿州 2023），可用
> `python scripts/fetch_real_samples.py --event all` 重新抓取。

⚠️ **这里的影像全部是程序合成的，不是真实卫星影像，只用于验证"全流程能跑通"。**

## 为什么用合成数据？

真实数据集 Sen1Floods11 有 4 GB+，下载、解压、格式对齐要几个小时；
而项目第一天就要能演示、能跑通训练链路。所以：

```powershell
python data/make_samples.py --n 6 --size 640
```

30 秒生成 6 景 Sentinel-2 风格影像 + 逐像元真值掩膜。

## 生成原理（不是随机噪声，有物理先验）

1. **地形**：多倍频分形噪声模拟高程，低洼处更容易被淹；
2. **地物光谱**：按水体/植被/农田/裸土/城市五类地物的典型反射率合成蓝、绿、红、近红外四波段；
3. **混合像元**：对分类掩膜做高斯模糊，模拟卫星影像边界处的光谱混合；
4. **洪水**：以"高程 + 距河道距离"打分，按分位数确定淹没区，沿河道漫入低洼区；
5. **传感器效应**：叠加大气雾与高斯噪声。

所以水体在 NDWI 上依然呈高值、城市依然是假阳性来源——**用它调算法是有意义的**。

## 文件命名

| 文件 | 含义 |
|---|---|
| `demoXX_pre.tif` | 灾前影像，4 波段（B2 蓝 / B3 绿 / B4 红 / B8 近红外），uint16，反射率 ×10000 |
| `demoXX_post.tif` | 灾后影像（含洪水淹没） |
| `demoXX_mask.png` | 灾后水体真值掩膜，白色=水（>127 视为水） |
| `demoXX_pre_mask.png` | 灾前水体真值（河道 + 湖泊） |
| `demoXX_preview.jpg` | 灾前/灾后真彩预览，横向拼接 |
| `samples.json` | 清单：面积、分辨率、描述 |

- 坐标系：EPSG:32650（UTM 50N），像元 10 m
- 单景 640×640 像元 = 6.4 km × 6.4 km ≈ 40.96 km²

## 什么时候必须换成真实数据？

**任何对外公布的精度指标，都必须来自真实数据。** 合成数据的 IoU 只能说明"代码没写错"，
不能说明"模型在真实洪灾上有效"。

```powershell
python scripts/download_data.py --dataset sen1floods11   # 看下载指引
python scripts/download_data.py --dataset sen1floods11 --check   # 检查是否就绪
```

整理成 `data/sen1floods11/xxx_post.tif + xxx_mask.png` 后即可训练：

```powershell
python train.py --data data/sen1floods11 --epochs 60 --img-size 512 --arch smp --encoder resnet34
```
