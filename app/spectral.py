"""Local multi-temporal monitoring UI, backed by the same CLI implementation."""
from pathlib import Path
import uuid
import zipfile

from src.spectral_monitor import run_monitor, format_summary_text


def run_spectral_ui(images, index, dates, band_order, min_valid_pct, out_root):
    if not images:
        raise ValueError("请提供按时间先后排列的本地 GeoTIFF")
    parsed_dates = str(dates or "").replace(",", " ").split() or None
    directory = Path(out_root) / ("spectral_" + uuid.uuid4().hex)
    summary = run_monitor([str(p) for p in images], index, str(directory),
                          dates=parsed_dates, band_order=band_order,
                          min_valid_pct=float(min_valid_pct))
    archive = directory / "spectral_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(directory.iterdir()):
            if path.is_file() and path != archive:
                zf.write(path, path.name)
    return format_summary_text(summary), str(archive)


def build_spectral_ui(out_root):
    import gradio as gr
    with gr.Accordion("多时相遥感监测 · 植被、水分、火烧迹地与地表水", open=False):
        gr.Markdown("按时间顺序上传 GeoTIFF，自动对齐网格并输出逐期指数、相邻差值及质量摘要。指数变化需结合现场资料解释。")
        images = gr.File(label="按时间先后排列的 GeoTIFF", file_count="multiple",
                         file_types=[".tif", ".tiff"], type="filepath")
        with gr.Row():
            index = gr.Dropdown(choices=[("植被绿度 NDVI", "ndvi"), ("稀疏植被 SAVI", "savi"),
                ("地表水 NDWI", "ndwi"), ("城市/裸地水体 MNDWI", "mndwi"),
                ("植被水分 NDMI", "ndmi"), ("火烧敏感 NBR", "nbr")], value="ndvi", label="监测方向")
            order = gr.Dropdown(choices=["auto", "s2_bgr_nir", "s2_6band", "s2_10band", "s2_l2a_12", "s2_l2a_13"],
                                value="auto", label="波段排列")
        dates = gr.Textbox(label="观测日期（可选；与文件逐一对应）", placeholder="2026-06-01 2026-07-01")
        valid = gr.Slider(0, 100, value=50, step=1, label="最低有效观测比例（%）")
        gr.Markdown("MNDWI / NDMI / NBR 需要真实短波红外。日期不填则标记未核验。所有差值为后减前，NBR 差值不能直接套用 dNBR 等级。网页上传不含 SCL 侧车；有质量侧车的数据建议使用本地命令行。")
        run = gr.Button("运行光谱监测")
        summary = gr.Textbox(label="结果与质量说明", lines=12)
        download = gr.File(label="GIS 栅格与摘要成果包")
        run.click(lambda *args: run_spectral_ui(*args, out_root=out_root),
                  inputs=[images, index, dates, order, valid], outputs=[summary, download],
                  api_name="run_spectral")
