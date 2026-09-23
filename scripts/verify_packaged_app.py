"""
慧眼识灾 · 打包产物验收脚本
============================

打包完 exe 之后，别只看"能不能启动"——这个脚本会自动：

    1. 以 headless 模式启动 exe（不弹窗口）
    2. 轮询端口直到服务就绪
    3. 通过 Gradio 官方客户端真实调用三个接口：
          · 单时相识别（真实影像）
          · 灾前灾后对比
          · 合成样本识别
    4. 检查成果包里 PDF/PNG/JSON 是否齐全
    5. 关闭进程，打印通过/失败清单

用法：
    python scripts/verify_packaged_app.py
    python scripts/verify_packaged_app.py --exe "dist_full/慧眼识灾/慧眼识灾.exe" --port 7873
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import zipfile
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass


def wait_port(port: int, timeout: float = 180.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(2.0)
    return False


def check(cond: bool, name: str, detail: str = "") -> bool:
    print(f"  [{'✓' if cond else '✗'}] {name}" + (f"  {detail}" if detail else ""))
    return cond


def main() -> int:
    ap = argparse.ArgumentParser(description="验收打包后的桌面应用")
    ap.add_argument("--exe", default=os.path.join(ROOT, "dist", "慧眼识灾", "慧眼识灾.exe"))
    ap.add_argument("--port", type=int, default=7873)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--keep-running", action="store_true", help="验收后不关闭进程")
    args = ap.parse_args()

    exe = os.path.abspath(args.exe)
    if not os.path.isfile(exe):
        print(f"✗ 找不到 exe：{exe}\n  先执行：powershell -ExecutionPolicy Bypass -File build/build_app.ps1 -Profile lite")
        return 2

    print("=" * 70)
    print(f"验收打包产物：{exe}")
    print(f"体积：{os.path.getsize(exe) / 1024 / 1024:.1f} MB（exe 本体）")
    print("=" * 70)

    workdir = os.path.dirname(exe)
    proc = subprocess.Popen(
        [exe, "--headless", "--port", str(args.port)],
        cwd=workdir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    results: List[bool] = []
    try:
        print(f"\n[1] 启动 exe（headless :{args.port}）…")
        ok = wait_port(args.port, args.timeout)
        results.append(check(ok, "服务端口就绪", f"耗时 ≤ {args.timeout:.0f}s"))
        if not ok:
            log = os.path.join(workdir, "logs", "desktop.log")
            if os.path.isfile(log):
                print("\n--- desktop.log 末尾 ---")
                # 用 with 关闭句柄（原实现 open(...).read() 会留一个未关闭的文件对象）
                with open(log, encoding="utf-8", errors="replace") as fh:
                    tail = fh.read().splitlines()[-15:]
                print("\n".join(tail))
            return 1

        from gradio_client import Client

        client = Client(f"http://127.0.0.1:{args.port}", verbose=False)

        print("\n[2] 单时相识别（真实影像 鄱阳湖 2020）")
        out = client.predict(None, None, "poyang2020", "NDWI + Otsu 基线", 10.0, 120, 0.5, False,
                             "（自动选择最新权重）", api_name="/run_single")
        summary, zip_path, stats_html = out[3], out[4], out[2]
        results.append(check("km²" in summary, "返回识别结论", summary.splitlines()[0][:46]))
        results.append(check("heye-hero" in stats_html, "统计面板渲染"))
        results.append(check(os.path.isfile(zip_path), "成果包已生成", os.path.basename(zip_path)))
        if os.path.isfile(zip_path):
            with zipfile.ZipFile(zip_path) as zf:
                names = set(zf.namelist())
                need = {"water_mask.png", "overlay.png", "true_color.png",
                        "probability_heatmap.png", "stats.json", "report.pdf"}
                results.append(check(need <= names, "成果包内容齐全", f"{len(names)} 个文件"))
                if "stats.json" in names:
                    payload = json.loads(zf.read("stats.json").decode("utf-8"))
                    # detail 参数会先求值：直接下标取值在缺键时抛 KeyError，
                    # 验收脚本会崩溃而不是标记该项失败。
                    area = payload.get("stats", {}).get("water_area_km2")
                    results.append(check(
                        area is not None and float(area) > 0,
                        "stats.json 数值有效",
                        f"水体 {float(area):.2f} km²" if area is not None else "字段缺失 water_area_km2",
                    ))
                if "report.pdf" in names:
                    pdf = zf.read("report.pdf")
                    results.append(check(pdf[:4] == b"%PDF" and len(pdf) > 5000,
                                         "PDF 简报有效", f"{len(pdf) // 1024} KB"))

        print("\n[3] 灾前灾后对比（涿州 2023）")
        out2 = client.predict(None, None, "zhuozhou2023", "NDWI + Otsu 基线", 10.0, 120, 0.5,
                              "（自动选择最新权重）", api_name="/run_compare")
        results.append(check("新增淹没" in out2[3], "返回双时相结论", out2[3].splitlines()[-1][:46]))
        results.append(check(bool(out2[1]), "变化检测图已生成"))

        print("\n[4] 合成样本（基线）")
        out3 = client.predict(None, None, "demo04", "NDWI + Otsu 基线", 10.0, 120, 0.5, False,
                              "（自动选择最新权重）", api_name="/run_single")
        results.append(check("km²" in out3[3], "合成样本识别", out3[3].splitlines()[0][:46]))

    finally:
        if not args.keep_running:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()
            print("\n已关闭 exe 进程")

    passed = sum(1 for r in results if r)
    print("\n" + "=" * 70)
    print(f"验收结果：{passed}/{len(results)} 项通过")
    print("=" * 70)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
