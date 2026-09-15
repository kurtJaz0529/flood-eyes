# -*- coding: utf-8 -*-
"""
record_demo.py — 自动录制「慧眼识灾」Gradio 演示视频（Playwright + Chromium）。

流程：
    首页停顿 -> 单景识别（poyang2020）-> 拖动对比滑块 -> 滚动统计面板
    -> 灾前/灾后对比（poyang2020）-> 拖动滑块 -> 滚动变化检测图/统计
    -> 切回首页定格 -> 保存 webm -> （有 ffmpeg 则）转 mp4

运行：
    python scripts/record_demo.py
输出：
    docs/assets/demo_auto_poyang.webm / .mp4
    docs/assets/rec_frames/*.png   关键步骤截图
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "docs", "assets")
FRAMES = os.path.join(ASSETS, "rec_frames")
VIDEO_TMP = os.path.join(ASSETS, "rec_video_tmp")
URL = "http://127.0.0.1:7860/"

W, H = 1920, 1080
ZOOM = 1.25          # CSS zoom，放大字体
TARGET_MIN_S = 65.0  # 不足则补静止帧，保证总时长在 60-90s 目标区间

FFMPEG_FALLBACK = (r"C:\Users\21679\AppData\Local\Microsoft\WinGet\Packages"
                   r"\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe"
                   r"\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe")

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def shot(page, name: str) -> None:
    path = os.path.join(FRAMES, f"{name}.png")
    try:
        page.screenshot(path=path)
        log(f"screenshot -> {name}.png")
    except Exception as exc:
        log(f"screenshot FAILED ({name}): {exc}")


def find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    if os.path.isfile(FFMPEG_FALLBACK):
        return FFMPEG_FALLBACK
    return None


# ----------------------------------------------------------------------
# 交互辅助
# ----------------------------------------------------------------------

def select_poyang(page, dd_label: str) -> None:
    """打开指定 aria-label 的 Gradio 6 下拉框并选择 poyang2020（鄱阳湖）。"""
    dd = page.get_by_role("combobox", name=dd_label)
    dd.wait_for(state="visible", timeout=15000)
    dd.click()
    page.wait_for_timeout(900)
    opt = page.get_by_role("option", name=re.compile("鄱阳湖"))
    opt.wait_for(state="visible", timeout=10000)
    opt.click()
    page.wait_for_timeout(900)
    val = dd.input_value()
    assert "鄱阳湖" in val, f"dropdown value unexpected: {val!r}"
    log(f"selected sample: {val}")


def count_done_textareas(page) -> int:
    return page.evaluate(
        "() => [...document.querySelectorAll('textarea')]"
        ".filter(t => (t.value || '').includes('耗时')).length"
    )


def wait_run_done(page, baseline: int, timeout_s: float) -> None:
    """等待 textarea 日志中出现新的「耗时」（完成标志）；同时侦测错误 toast。"""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        n = count_done_textareas(page)
        if n > baseline:
            log(f"run finished (done-logs {baseline} -> {n})")
            return
        err = page.evaluate(
            "() => [...document.querySelectorAll('.toast-body, [role=\"alert\"]')]"
            ".filter(e => e.offsetParent !== null).map(e => e.textContent).join(' | ')"
        )
        if err and ("失败" in err or "Error" in err):
            raise RuntimeError(f"Gradio error toast: {err[:200]}")
        page.wait_for_timeout(600)
    raise TimeoutError(f"run not finished within {timeout_s}s")


def click_run_button(page, text: str) -> int:
    """点击运行按钮，返回点击前的完成日志计数（作为 wait_run_done 基线）。"""
    baseline = count_done_textareas(page)
    btn = page.locator(f'button:has-text("{text}"):visible').first
    btn.wait_for(state="visible", timeout=15000)
    btn.click()
    log(f"clicked button: {text}")
    return baseline


def drag_slider(page, label_part: str, tag: str) -> None:
    """在 ImageSlider 上缓慢来回拖动一次（30% <-> 70% 宽度）。"""
    block = page.locator(
        f'xpath=//*[contains(text(),"{label_part}")]'
        f'/ancestor::div[contains(@class,"block")][.//img][1]'
    )
    if block.count() == 0:
        block = page.locator(f'div.block:has-text("{label_part}")').last
    block.wait_for(state="visible", timeout=15000)
    block.scroll_into_view_if_needed()
    page.wait_for_timeout(600)

    img = block.locator("img").first
    box = img.bounding_box() if img.count() else None
    if not box:
        box = block.bounding_box()
    if not box:
        raise RuntimeError(f"slider block has no bounding box: {label_part}")

    y = box["y"] + box["height"] * 0.5
    x_lo = box["x"] + box["width"] * 0.30
    x_hi = box["x"] + box["width"] * 0.70
    log(f"drag slider '{label_part}' y={y:.0f} x:[{x_lo:.0f} -> {x_hi:.0f}]")

    steps, dwell = 26, 0.07
    page.mouse.move(x_lo, y)
    page.wait_for_timeout(250)
    page.mouse.down()
    for i in range(1, steps + 1):  # 30% -> 70%
        page.mouse.move(x_lo + (x_hi - x_lo) * i / steps, y)
        page.wait_for_timeout(int(dwell * 1000))
    page.wait_for_timeout(700)
    shot(page, f"{tag}_drag_mid")
    for i in range(1, steps + 1):  # 70% -> 30%
        page.mouse.move(x_hi - (x_hi - x_lo) * i / steps, y)
        page.wait_for_timeout(int(dwell * 1000))
    page.mouse.up()
    page.wait_for_timeout(600)
    log("slider drag done")


def smooth_scroll(page, dy: int, times: int = 3) -> None:
    step = dy // times
    for _ in range(times):
        page.evaluate(f"window.scrollBy({{top: {step}, behavior: 'smooth'}})")
        page.wait_for_timeout(500)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------

def main() -> int:
    os.makedirs(FRAMES, exist_ok=True)
    os.makedirs(VIDEO_TMP, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=VIDEO_TMP,
            record_video_size={"width": W, "height": H},
            locale="zh-CN",
        )
        page = ctx.new_page()
        video = page.video

        try:
            # ---------- 首页 ----------
            log(f"open {URL}")
            page.goto(URL, wait_until="domcontentloaded")
            page.wait_for_selector("text=慧眼识灾", timeout=30000)
            page.evaluate(f"document.body.style.zoom = '{ZOOM}'")
            page.wait_for_timeout(3000)
            shot(page, "01_home")

            # ---------- 页签1：单景识别 ----------
            select_poyang(page, "示例样本（无数据也能演示）")
            page.wait_for_timeout(1200)  # 统计预览出现
            shot(page, "02_sample_selected")

            base = click_run_button(page, "开始识别")
            wait_run_done(page, base, timeout_s=120)
            page.wait_for_timeout(1800)
            shot(page, "03_single_result")

            drag_slider(page, "拖动滑块：原图", "04_single")

            # 滚动展示统计面板 / 结论区
            smooth_scroll(page, 660, times=3)
            page.get_by_text("识别结论", exact=False).first.scroll_into_view_if_needed()
            page.wait_for_timeout(3200)
            shot(page, "05_single_stats")
            page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
            page.wait_for_timeout(900)

            # ---------- 页签2：灾前/灾后对比 ----------
            page.get_by_role("tab", name=re.compile("灾前")).click()
            page.wait_for_timeout(1500)
            shot(page, "06_tab2")
            select_poyang(page, "示例样本（同一景的灾前 + 灾后）")
            page.wait_for_timeout(900)
            shot(page, "07_tab2_sample")

            base = click_run_button(page, "开始对比")
            wait_run_done(page, base, timeout_s=180)
            page.wait_for_timeout(1800)
            shot(page, "08_compare_result")

            drag_slider(page, "拖动滑块：灾前", "09_compare")

            smooth_scroll(page, 620, times=3)
            page.get_by_text("对比结论", exact=False).first.scroll_into_view_if_needed()
            page.wait_for_timeout(3200)
            shot(page, "10_compare_stats")

            # ---------- 收尾：切回页签1 定格 ----------
            page.get_by_role("tab", name=re.compile("单时相")).click()
            page.wait_for_timeout(600)
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(2500)
            shot(page, "11_final")

            # 补足最小时长
            elapsed = time.time() - T0
            if elapsed < TARGET_MIN_S:
                pad = TARGET_MIN_S - elapsed
                log(f"pad {pad:.1f}s to reach {TARGET_MIN_S:.0f}s total")
                page.wait_for_timeout(int(pad * 1000))

        except Exception as exc:
            log(f"ERROR: {type(exc).__name__}: {exc}")
            try:
                shot(page, "99_error")
                dump = page.evaluate("() => document.body ? document.body.innerText.slice(0, 3000) : ''")
                with open(os.path.join(FRAMES, "error_dump.txt"), "w", encoding="utf-8") as f:
                    f.write(dump)
            except Exception:
                pass
            ctx.close()
            browser.close()
            return 1

        ctx.close()   # 关闭 context 后视频才落盘
        src_video = video.path()  # 必须在退出 sync_playwright 之前取路径
        browser.close()

    # ---------- 保存视频 ----------
    webm_out = os.path.join(ASSETS, "demo_auto_poyang.webm")
    shutil.copyfile(src_video, webm_out)
    log(f"webm saved: {webm_out} ({os.path.getsize(webm_out)/1e6:.2f} MB)")

    ffmpeg = find_ffmpeg()
    if ffmpeg:
        mp4_out = os.path.join(ASSETS, "demo_auto_poyang.mp4")
        cmd = [ffmpeg, "-y", "-i", webm_out, "-c:v", "libx264", "-preset", "medium",
               "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", mp4_out]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode == 0:
            log(f"mp4 saved: {mp4_out} ({os.path.getsize(mp4_out)/1e6:.2f} MB)")
        else:
            log(f"ffmpeg failed:\n{r.stderr[-800:]}")
    else:
        log("ffmpeg not found; keeping webm only")

    shutil.rmtree(VIDEO_TMP, ignore_errors=True)
    log(f"TOTAL {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
