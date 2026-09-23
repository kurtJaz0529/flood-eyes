# -*- coding: utf-8 -*-
"""生成「慧眼识灾」项目路演 PPT（16:9，13页）"""
import os
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

NAVY = RGBColor(0x0B, 0x3D, 0x62)
BLUE = RGBColor(0x1B, 0x7F, 0xB8)
CYAN = RGBColor(0x17, 0xA2, 0xB8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY = RGBColor(0x44, 0x44, 0x44)
LIGHT = RGBColor(0xF2, 0xF7, 0xFA)
RED = RGBColor(0xC0, 0x39, 0x2B)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = prs.slide_width, prs.slide_height

def slide():
    return prs.slides.add_slide(BLANK)

def rect(s, x, y, w, h, color, line=False):
    from pptx.enum.shapes import MSO_SHAPE
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = color
    if line:
        sh.line.color.rgb = BLUE; sh.line.width = Pt(1)
    else:
        sh.line.fill.background()
    return sh

def text(s, x, y, w, h, lines, size=18, color=GRAY, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line_spacing=1.15):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    tf.vertical_anchor = anchor
    if isinstance(lines, str): lines = [(lines, size, color, bold)]
    for i, (t, sz, c, b) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.line_spacing = line_spacing
        r = p.add_run(); r.text = t
        r.font.size = Pt(sz); r.font.color.rgb = c; r.font.bold = b
        r.font.name = "微软雅黑"
    return tb

def header(s, title, subtitle=None):
    rect(s, 0, 0, W, Inches(0.9), NAVY)
    rect(s, 0, Inches(0.9), W, Pt(3), CYAN)
    text(s, Inches(0.5), Inches(0.12), Inches(11), Inches(0.7),
         [(title, 26, WHITE, True)])
    if subtitle:
        text(s, Inches(0.5), Inches(1.05), Inches(12), Inches(0.4),
             [(subtitle, 13, BLUE, False)])

def bullets(s, x, y, w, h, items, size=16, gap=True):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.25
        p.space_after = Pt(10 if gap else 4)
        if isinstance(it, tuple):
            head, body = it
            r = p.add_run(); r.text = "▍" + head
            r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = "微软雅黑"
            if body:
                r2 = p.add_run(); r2.text = "  " + body
                r2.font.size = Pt(size - 1); r2.font.color.rgb = GRAY; r2.font.name = "微软雅黑"
        else:
            r = p.add_run(); r.text = "• " + it
            r.font.size = Pt(size); r.font.color.rgb = GRAY; r.font.name = "微软雅黑"
    return tb

def pic(s, path, x, y, w=None, h=None):
    # 图片缺失时 add_picture 会直接抛异常，整份 PPT 白做；先给出明确提示
    if not os.path.isfile(path):
        raise SystemExit(f"缺少配图：{path}\n请先跑对应流程生成图片，或用 --assets 指定资源目录")
    return s.shapes.add_picture(path, x, y, width=w, height=h)

# 资源与产物路径全部由项目根目录推导（原实现写死 F:/deepseek/... 与
# C:\Users\<用户名>\Desktop\...，换机器必然失败）。
A = os.environ.get("HUIYAN_PPT_ASSETS") or os.path.join(ROOT, "docs", "assets")
_out_env = os.environ.get("HUIYAN_PPT_OUT")
OUT = _out_env or os.path.join(ROOT, "docs", "慧眼识灾-项目路演PPT.pptx")


def _latest_output_dir(prefix: str) -> str:
    """在 outputs/ 下找最近一次生成的成果目录，避免写死带时间戳的目录名。"""
    root = os.path.join(ROOT, "outputs")
    if not os.path.isdir(root):
        return ""
    cands = [
        os.path.join(root, name)
        for name in os.listdir(root)
        if name.startswith(prefix) and os.path.isdir(os.path.join(root, name))
    ]
    return sorted(cands, key=os.path.getmtime, reverse=True)[0] if cands else ""


# 兼容旧变量名：找不到时为空串，取图处会给出明确报错
O_PY = _latest_output_dir("poyang2020_post.tif")
O_ZZ = _latest_output_dir("zhuozhou2023_post.tif")

# ---------- 1 封面 ----------
s = slide()
rect(s, 0, 0, W, H, NAVY)
rect(s, 0, Inches(5.9), W, Pt(3), CYAN)
text(s, Inches(1), Inches(1.6), Inches(11.3), Inches(1.6),
     [("慧眼识灾", 54, WHITE, True)], align=PP_ALIGN.CENTER)
text(s, Inches(1), Inches(3.0), Inches(11.3), Inches(0.8),
     [("基于深度学习的遥感洪水智能识别系统", 30, CYAN, True)], align=PP_ALIGN.CENTER)
text(s, Inches(1), Inches(4.2), Inches(11.3), Inches(1.2),
     [("把灾后人工目视解译的 1–3 天，压缩到分钟级自动完成", 16, WHITE, False),
      ("中国国际大学生创新大赛（2026）· 高教主赛道 · 本科生创意组 · “人工智能+”", 14, RGBColor(0x9F,0xC5,0xE8), False)],
     align=PP_ALIGN.CENTER)
text(s, Inches(1), Inches(6.3), Inches(11.3), Inches(0.8),
     [("项目负责人：郝佳鑫    团队成员：宋艺欣    地理与规划学院 · 遥感科学与技术 2025级", 14, WHITE, False)],
     align=PP_ALIGN.CENTER)

# ---------- 2 痛点 ----------
s = slide(); header(s, "01  背景与痛点", "洪水应急，时间就是生命")
bullets(s, Inches(0.6), Inches(1.5), Inches(6.4), Inches(5),
        [("灾情就是命令", "洪灾发生后，首要问题是：哪里被淹了？淹了多少？"),
         ("人工解译太慢", "传统依赖专家目视解译卫星影像，一圈一画需要 1–3 天"),
         ("错过黄金救援期", "72 小时黄金窗口内，迟到的灾情图 = 滞后的救援决策"),
         ("基层缺工具", "应急、水利基层部门缺少开箱即用的自动化分析工具")], size=17)
rect(s, Inches(7.4), Inches(1.6), Inches(5.4), Inches(4.6), LIGHT)
text(s, Inches(7.7), Inches(1.9), Inches(4.8), Inches(4),
     [("1–3 天", 44, RED, True),
      ("传统人工目视解译一景影像的耗时", 14, GRAY, False),
      ("", 10, GRAY, False),
      ("分钟级", 44, CYAN, True),
      ("本系统单景自动识别耗时（CPU）", 14, GRAY, False)])

# ---------- 3 解决方案 ----------
s = slide(); header(s, "02  解决方案", "慧眼识灾：端到端洪水智能识别流水线")
steps = [("卫星影像", "Sentinel-2\n公开免费"), ("智能预处理", "波段计算\nBOA偏移校正"),
         ("双引擎识别", "U-Net 深度分割\n+ NDWI 物理基线"), ("自动统计", "淹没面积\n变化分析"),
         ("一键简报", "成果包\nPDF报告")]
x = Inches(0.45)
for i, (t1, t2) in enumerate(steps):
    rect(s, x, Inches(2.2), Inches(2.3), Inches(1.7), LIGHT, line=True)
    text(s, x, Inches(2.45), Inches(2.3), Inches(1.3),
         [(t1, 18, NAVY, True), (t2, 12, GRAY, False)], align=PP_ALIGN.CENTER)
    if i < 4:
        text(s, x + Inches(2.3), Inches(2.7), Inches(0.35), Inches(0.6),
             [("▶", 20, CYAN, True)], align=PP_ALIGN.CENTER)
    x += Inches(2.62)
bullets(s, Inches(0.6), Inches(4.5), Inches(12.2), Inches(2.5),
        [("全自动", "上传影像 → 圈定淹没范围 → 算出受灾面积 → 导出简报，全程无需人工干预"),
         ("双引擎互证", "深度学习与物理水体指数两条独立路线交叉验证，结果可信、可解释"),
         ("真实验证", "已在 2020 鄱阳湖特大洪水、2023 涿州暴雨真实卫星影像上完成验证")], size=16)

# ---------- 4 系统演示 ----------
s = slide(); header(s, "03  系统演示", "Web 系统 · 上传即识别 · 现场可演示")
pic(s, f"{A}/rec_frames/03_single_result.png", Inches(0.5), Inches(1.4), w=Inches(7.6))
bullets(s, Inches(8.4), Inches(1.6), Inches(4.5), Inches(5),
        [("三个功能页签", "单景识别 / 灾前灾后对比 / 系统说明"),
         ("原图↔结果滑块", "拖动对比，识别效果一目了然"),
         ("统计面板", "面积、占比、连通水域自动计算"),
         ("成果包导出", "掩膜/叠加图/热力图/PDF 简报打包下载")], size=15)
text(s, Inches(8.4), Inches(5.9), Inches(4.5), Inches(0.8),
     [("▶ 路演现场联网演示：http://127.0.0.1:7860", 12, BLUE, False)])

# ---------- 5 案例：鄱阳湖 ----------
s = slide(); header(s, "04  真实灾例验证：2020·江西鄱阳湖特大洪水", "Sentinel-2 真实影像 · 10m 分辨率 · 12.8km×12.8km 窗口")
pic(s, f"{O_PY}/before_after.png", Inches(0.45), Inches(1.5), w=Inches(7.9))
rect(s, Inches(8.6), Inches(1.6), Inches(4.3), Inches(4.4), LIGHT)
text(s, Inches(8.85), Inches(1.85), Inches(3.8), Inches(3.9),
     [("识别结果（U-Net）", 15, NAVY, True),
      ("灾后水体面积", 13, GRAY, False), ("155.23 km²", 26, CYAN, True),
      ("新增淹没面积", 13, GRAY, False), ("105.57 km²", 26, RED, True),
      ("单景识别耗时 2.9 秒（CPU）", 13, GRAY, False),
      ("2020-05-19 → 2020-07-15 洪峰期", 12, GRAY, False)])

# ---------- 6 交叉验证 ----------
s = slide(); header(s, "05  双引擎交叉验证", "两条独立技术路线，结果相互印证")
rows = [("指标", "NDWI+Otsu 物理基线", "U-Net 深度模型", "偏差"),
        ("灾后水体面积", "154.37 km²", "155.23 km²", "0.6%"),
        ("新增淹没面积", "110.72 km²", "105.57 km²", "4.7%")]
from pptx.util import Pt as _Pt
tbl = s.shapes.add_table(4, 4, Inches(0.9), Inches(1.8), Inches(11.5), Inches(2.6)).table
for i, row in enumerate(rows):
    for j, v in enumerate(row):
        c = tbl.cell(i, j); c.text = v
        for p in c.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(16); r.font.name = "微软雅黑"
                r.font.bold = (i == 0); r.font.color.rgb = WHITE if i == 0 else GRAY
        if i == 0:
            c.fill.solid(); c.fill.fore_color.rgb = NAVY
bullets(s, Inches(0.9), Inches(4.9), Inches(11.5), Inches(2),
        ["物理基线不依赖任何训练数据，是独立第三方的“裁判员”",
         "两路线结果偏差 < 5%：说明 AI 模型的识别结果科学可信",
         "更多场景：2023 河北涿州暴雨灾例同步完成系统测试"], size=15)

# ---------- 7 技术创新 ----------
s = slide(); header(s, "06  技术创新点")
bullets(s, Inches(0.6), Inches(1.5), Inches(12.2), Inches(5.5),
        [("双引擎架构", "U-Net 语义分割与 NDWI+Otsu 物理基线并行推理、互验互校，兼顾精度与可解释性"),
         ("遥感数据工程", "解决 Sentinel-2 L2A 产品 BOA −1000 偏移量坑（官方标记存在已知错误），与原始 JP2 逐窗比对自动判定；处理 UTM 瓦片旋转 nodata"),
         ("大影像切块推理", "512×512 滑动窗口 + 重叠拼接，支持任意尺寸影像"),
         ("端到端产品化", "识别→统计→变化分析→PDF 灾情简报一键生成，非Demo玩具，是可交付工具"),
         ("零样本迁移", "模型迁移到真实洪灾场景，与物理基线结果偏差 <5%")], size=17)

# ---------- 8 应用场景 ----------
s = slide(); header(s, "07  应用前景与商业模式")
bullets(s, Inches(0.6), Inches(1.5), Inches(6.2), Inches(5.5),
        [("应急管理", "灾中快速圈定淹没范围，辅助救援力量调度"),
         ("水利/自然资源", "河湖监管、行洪区监测、国土变更调查"),
         ("保险定损", "农业保险、财产险受灾面积快速核定"),
         ("科研教育", "为高校遥感课程提供实训平台")], size=17)
rect(s, Inches(7.2), Inches(1.6), Inches(5.6), Inches(4.8), LIGHT)
text(s, Inches(7.5), Inches(1.9), Inches(5), Inches(4.3),
     [("商业模式", 18, NAVY, True),
      ("· SaaS 订阅：按景/按区域计费", 14, GRAY, False),
      ("· 项目制：为地方应急部门定制部署", 14, GRAY, False),
      ("· 数据服务：灾情快报订阅推送", 14, GRAY, False),
      ("", 10, GRAY, False),
      ("数据成本为零", 18, NAVY, True),
      ("Sentinel-2 每 5 天免费重访，", 14, GRAY, False),
      ("持续供给最新影像", 14, GRAY, False)])

# ---------- 9 已有成果 ----------
s = slide(); header(s, "08  项目基础与已有成果", "不是纸上谈兵——系统已经跑起来了")
bullets(s, Inches(0.6), Inches(1.5), Inches(7.2), Inches(5.5),
        [("可运行系统", "Web 识别平台完整可用，2300+ 行代码，模块化设计"),
         ("真实灾例验证", "鄱阳湖 2020、涿州 2023 双案例全流程跑通"),
         ("自动化演示视频", "64 秒系统实操录像（Playwright 自动录制）"),
         ("完整溯源文档", "数据来源、处理方法、参数全部留痕可查"),
         ("规划", "申请软件著作权 1 项；发表学术论文 1 篇")], size=17)
pic(s, f"{O_PY}/overlay.png", Inches(8.3), Inches(1.5), h=Inches(4.6))
text(s, Inches(8.3), Inches(6.2), Inches(4.4), Inches(0.5),
     [("鄱阳湖灾后影像识别叠加图", 12, BLUE, False)], align=PP_ALIGN.CENTER)

# ---------- 10 团队 ----------
s = slide(); header(s, "09  团队介绍")
rect(s, Inches(0.6), Inches(1.6), Inches(5.9), Inches(4.6), LIGHT)
text(s, Inches(0.95), Inches(1.9), Inches(5.2), Inches(4),
     [("郝佳鑫 · 项目负责人", 20, NAVY, True),
      ("遥感科学与技术 2025级", 14, GRAY, False),
      ("项目统筹 / 应用场景设计 / 材料撰写", 14, GRAY, False),
      ("", 10, GRAY, False),
      ("宋艺欣 · 核心成员", 20, NAVY, True),
      ("遥感科学与技术 2025级", 14, GRAY, False),
      ("数据处理 / 系统测试 / 演示验证", 14, GRAY, False)])
rect(s, Inches(6.9), Inches(1.6), Inches(5.9), Inches(4.6), LIGHT)
text(s, Inches(7.25), Inches(1.9), Inches(5.2), Inches(4),
     [("指导教师：（待聘）", 20, NAVY, True),
      ("拟聘遥感/GIS 方向专业教师", 14, GRAY, False),
      ("", 10, GRAY, False),
      ("团队优势", 18, NAVY, True),
      ("· 地理信息+AI 交叉学科背景", 14, GRAY, False),
      ("· 全流程独立完成：数据/算法/产品", 14, GRAY, False),
      ("· 代码、实验记录全程留痕", 14, GRAY, False)])

# ---------- 11 规划 ----------
s = slide(); header(s, "10  发展规划")
plans = [("近期 · 省赛前", "真实标注数据集微调模型，补齐 IoU/F1 真实指标；GPU 加速训练；申请软著"),
         ("中期 · 半年", "接入更多灾例（滑坡、内涝）；接入国产高分影像；小范围试点部署"),
         ("远期 · 一年", "多灾种扩展；SaaS 平台上线；与地方应急部门合作落地")]
y = Inches(1.7)
for t1, t2 in plans:
    rect(s, Inches(0.8), y, Inches(2.6), Inches(1.35), NAVY)
    text(s, Inches(0.8), y + Inches(0.4), Inches(2.6), Inches(0.6),
         [(t1, 16, WHITE, True)], align=PP_ALIGN.CENTER)
    rect(s, Inches(3.6), y, Inches(8.9), Inches(1.35), LIGHT)
    text(s, Inches(3.9), y + Inches(0.35), Inches(8.4), Inches(0.9),
         [(t2, 14, GRAY, False)])
    y += Inches(1.7)

# ---------- 12 合规 ----------
s = slide(); header(s, "11  诚信与合规声明")
bullets(s, Inches(0.8), Inches(1.7), Inches(11.6), Inches(5),
        ["全部参赛材料由团队独立完成，无任何商业机构外包",
         "代码仓库提交历史、模型训练日志、实验记录完整保留，成员实质性贡献可查",
         "数据来源合法合规：Sentinel-2 为欧空局免费开放数据（可商用）",
         "真实影像验证结果均标注方法来源与适用边界，不夸大精度",
         "合成数据仅用于流程开发，所有对外指标均注明其数据基础"], size=18)

# ---------- 13 结尾 ----------
s = slide()
rect(s, 0, 0, W, H, NAVY)
text(s, Inches(1), Inches(2.6), Inches(11.3), Inches(1.2),
     [("慧眼识灾 · 分钟级响应", 40, WHITE, True)], align=PP_ALIGN.CENTER)
text(s, Inches(1), Inches(3.9), Inches(11.3), Inches(0.8),
     [("让每一小时的救援，都快人一步", 20, CYAN, False)], align=PP_ALIGN.CENTER)
text(s, Inches(1), Inches(5.2), Inches(11.3), Inches(0.6),
     [("恳请各位评委老师批评指正", 16, WHITE, False)], align=PP_ALIGN.CENTER)

out = OUT
# prs.save 不会自动建目录，目标目录不存在会直接抛异常
os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
prs.save(out)
# 用计数器代替 prs.slides 的私有属性（原写法依赖 python-pptx 内部实现，库升级即失效）
print("saved:", out, "slides:", len(prs.slides._sldIdLst))
