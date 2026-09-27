"""
慧眼识灾 · 桌面应用启动器
=========================

双击 exe 后发生的事情：

    1. 找一个可用端口（默认 7860，被占用就自动往后找）
    2. 在后台线程启动 Gradio 本地服务（只监听 127.0.0.1，不对外网暴露）
    3. 用 Edge/Chrome 的 --app 模式打开一个"没有地址栏的窗口"，看起来就是原生软件
       （找不到 Edge/Chrome 就退回默认浏览器）
    4. 弹一个小控制窗（打开界面 / 退出程序），用户点退出才真正结束进程

参数：
    --headless       只起服务不开窗口（自动化测试 / 服务器部署用）
    --no-window      不开应用窗口，只打印地址
    --port 7860      指定端口
    --host 127.0.0.1 监听地址（默认只本机可访问）
    --baseline-only  只暴露 NDWI 基线模型
    --control        强制显示控制窗口
    --version        打印版本
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # Windows 控制台默认 GBK，中文输出会报错
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

from src.paths import bundle_root, logs_dir, user_root, weights_dirs  # noqa: E402

APP_NAME = "慧眼识灾 · 遥感 AI 洪水识别系统"
try:  # 版本号单一来源：src/__init__.py，避免界面显示与安装包名不一致
    from src import __version__ as APP_VERSION  # noqa: E402
except Exception:  # pragma: no cover
    APP_VERSION = "0.5.0"


# --------------------------------------------------------------------------
# 基础设施
# --------------------------------------------------------------------------


def find_free_port(host: str, start: int, tries: int = 40) -> int:
    """从 start 开始找一个能绑定的端口。"""
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"{host}:{start}~{start + tries} 没有可用端口")


def setup_logging() -> str:
    """把输出同时写到文件，方便打包后（无控制台）排查问题。

    注意：替换 sys.stdout 必须提供 isatty/encoding/fileno 等接口，
    否则 uvicorn 的彩色日志格式化器会在启动时直接抛异常。
    """
    path = os.path.join(logs_dir(), "desktop.log")
    try:
        fh = open(path, "a", encoding="utf-8", buffering=1)
    except Exception:
        return ""
    # 进程结束前关闭句柄。_Tee.close 是 no-op（避免 Gradio 把日志文件关掉），
    # 若不再注册 atexit，这个句柄会一直挂到进程被强杀为止。
    import atexit

    atexit.register(fh.close)

    class _Tee:
        encoding = "utf-8"
        errors = "replace"

        def __init__(self, stream) -> None:
            self._fh = stream

        def write(self, data: str) -> int:
            for target in (sys.__stdout__, self._fh):
                if target is None:
                    continue
                try:
                    target.write(data)
                except Exception:
                    pass
            return len(data)

        def flush(self) -> None:
            for target in (sys.__stdout__, self._fh):
                if target is None:
                    continue
                try:
                    target.flush()
                except Exception:
                    pass

        def isatty(self) -> bool:
            return False

        def writable(self) -> bool:
            return True

        def fileno(self) -> int:
            raise OSError("fileno 不可用")

        @property
        def closed(self) -> bool:
            return False

        def close(self) -> None:  # 不让 Gradio/uvicorn 把日志文件关掉
            pass

    sys.stdout = _Tee(fh)  # type: ignore[assignment]
    sys.stderr = sys.stdout  # type: ignore[assignment]
    return path


def bypass_proxy_for_localhost() -> None:
    """让本机回环地址绕过代理。

    Gradio 启动时会用 httpx（默认 trust_env=True）请求自己的
    ``http://127.0.0.1:<port>/startup-events``。如果机器上存在 HTTP_PROXY /
    HTTPS_PROXY（企业代理、容器、IDE 沙箱注入的残留变量），这个本机请求会被
    送到代理去转发，代理拒绝或超时后 launch() 直接抛 ConnectTimeout ——
    用户看到的现象是"双击了没反应"，日志里只有一行权重提示。

    只把回环地址加入 NO_PROXY，不改动用户代理；在线地图与卫星下载仍按用户
    原有代理设置出网。
    """
    parts = []
    for var in ("NO_PROXY", "no_proxy"):
        for entry in os.environ.get(var, "").split(","):
            entry = entry.strip()
            if entry and entry not in parts:
                parts.append(entry)
    for host in ("127.0.0.1", "localhost", "::1"):
        if host not in parts:
            parts.append(host)
    for var in ("NO_PROXY", "no_proxy"):
        os.environ[var] = ",".join(parts)


def start_startup_watchdog(deadline_sec: float = 60.0) -> threading.Event:
    """超时记录各线程堆栈；单凭超时不推断安全软件或网络根因。"""
    done = threading.Event()

    def _warn() -> None:
        if done.is_set():
            return
        print(
            f"[启动诊断] {deadline_sec:.0f} 秒内尚未完成启动。以下为当前线程调用栈；\n"
            "请结合具体异常排查端口、文件权限、依赖加载或本机连接情况。",
            flush=True,
        )
        import traceback
        for thread_id, frame in sys._current_frames().items():
            print(f"[线程 {thread_id}]", flush=True)
            traceback.print_stack(frame, file=sys.stderr)

    timer = threading.Timer(deadline_sec, _warn)
    timer.daemon = True
    timer.start()
    return done


FIREWALL_RULE_NAME = "慧眼识灾 本机界面"


class LoopbackError(OSError):
    """本机回环自检失败。继承 OSError，兼容既有的 `except OSError` 调用方。"""

    def __init__(self, stage: str, cause: str, detail: str, remedy: str, original: BaseException):
        super().__init__(f"{stage}: {type(original).__name__}: {original}")
        self.stage = stage
        self.cause = cause
        self.detail = detail
        self.remedy = remedy
        self.original = original

    def as_dict(self) -> dict:
        return {"status": "failed", "stage": self.stage, "cause": self.cause,
                "detail": self.detail, "remedy": self.remedy,
                "error": f"{type(self.original).__name__}: {self.original}"}


def firewall_remedy(exe_path: str, rule_name: str = FIREWALL_RULE_NAME) -> str:
    """放行本程序入站连接所需的命令（需要管理员权限的命令行）。

    只放行本程序与 TCP：界面是本机 HTTP 服务，用不到 UDP，也无需对其他程序开口。
    """
    return (f'netsh advfirewall firewall add rule name="{rule_name}" dir=in action=allow '
            f'protocol=TCP program="{exe_path}" localip=127.0.0.1 remoteip=127.0.0.1 '
            'enable=yes profile=any')


def classify_loopback_failure(stage: str, exc: BaseException, exe_path: str) -> LoopbackError:
    """报告可观察到的失败，不把超时直接认定为某一种防火墙。

    改名副本与原程序的差异支持按进程过滤的可能性，不能识别具体过滤组件。
    """
    remedy = firewall_remedy(exe_path)
    if isinstance(exc, TimeoutError) and stage in ("connect", "accept"):
        return LoopbackError(
            stage, "loopback_timeout",
            "本机回环连接超时，可能与防火墙、安全软件或其他进程网络策略有关。"
            "仅凭超时不能确定过滤方向或具体拦截组件，需结合系统日志排查。",
            remedy, exc)
    if isinstance(exc, ConnectionRefusedError):
        return LoopbackError(
            stage, "not_listening",
            "本机端口拒绝连接，说明该端口上没有监听者，属于程序内部状态异常，"
            "不是网络策略问题。", "", exc)
    if isinstance(exc, PermissionError):
        return LoopbackError(
            stage, "bind_denied",
            "绑定本机端口被拒绝，通常是端口被占用或权限受限。", "", exc)
    return LoopbackError(
        stage, "unknown",
        "本机回环自检出现未归类错误，请结合原始异常排查。", "", exc)


def check_loopback(timeout: float = 4.0) -> dict:
    """启动界面前验证回环连接，避免 asyncio 的 socketpair 无限等待。

    Windows 上 ``socket.socketpair()`` 是用回环 TCP 对模拟的，且没有超时；
    一旦入站连接被过滤，``asyncio`` 建事件循环时会永久阻塞，界面永远起不来。
    所以必须在建循环之前先把这一步探明。
    """
    exe_path = sys.executable
    listener = client = None
    stage = "bind"
    try:
        listener = socket.socket()
        client = socket.socket()
        listener.settimeout(timeout)
        client.settimeout(timeout)
        listener.bind(("127.0.0.1", 0))
        stage = "listen"
        listener.listen(1)
        stage = "connect"
        client.connect(listener.getsockname())
        stage = "accept"
        peer, _ = listener.accept()
        try:
            peer.settimeout(timeout)
            peer.sendall(b"ok")
            stage = "recv"
            if client.recv(2) != b"ok":
                raise ConnectionError("本机连接校验数据不完整")
        finally:
            peer.close()
    except OSError as exc:
        raise classify_loopback_failure(stage, exc, exe_path) from exc
    finally:
        for sock in (client, listener):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
    return {"status": "ok", "address": "127.0.0.1"}


def loopback_failure_message(exc: BaseException, log_path: Optional[str]) -> str:
    """把回环自检失败整理成给用户看的说明（含可执行的解决办法）。"""
    lines = [f"无法建立本机连接（127.0.0.1）：{type(exc).__name__}: {exc}"]
    if isinstance(exc, LoopbackError):
        lines += ["", f"判定：{exc.detail}"]
        if exc.remedy:
            lines += ["",
                      "如确认是 Windows 防火墙拦截，可选择以下一种方式尝试（仅放行回环）：",
                      "  1. 右键以管理员身份运行 _internal\\scripts\\allow_loopback.ps1，"
                      "为本程序放行入站连接；",
                      "  2. 在管理员命令行执行：",
                      f"     {exc.remedy}"]
    lines += ["", "界面服务未启动。", f"日志：{log_path}"]
    return "\n".join(lines)


def find_browser_app() -> Optional[str]:
    """找 Edge / Chrome 的可执行文件（用于 --app 无边框窗口模式）。"""
    candidates = [
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def open_app_window(url: str, width: int = 1440, height: int = 920) -> str:
    """优先用 Edge/Chrome 的 --app 模式（无地址栏，像原生软件），否则退回默认浏览器。"""
    exe = find_browser_app()
    if exe:
        try:
            subprocess.Popen(
                [exe, f"--app={url}", f"--window-size={width},{height}", "--disable-extensions"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return "app-window"
        except Exception as exc:
            print(f"[warn] 打开应用窗口失败({exc})，改用默认浏览器")
    webbrowser.open(url)
    return "browser"


def control_window(url: str, on_quit) -> None:
    """一个小控制窗：显示地址 + 打开界面 + 退出程序（tkinter，标准库自带）。"""
    try:
        import tkinter as tk
    except Exception as exc:  # 无 tkinter 就退化为命令行等待
        print(f"[warn] 无法创建控制窗口({exc})，按 Ctrl+C 退出")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            on_quit()
        return

    root = tk.Tk()
    root.title(f"{APP_NAME}")
    root.geometry("460x200")
    root.resizable(False, False)
    try:
        root.attributes("-topmost", False)
    except Exception:
        pass

    frame = tk.Frame(root, padx=18, pady=14)
    frame.pack(fill="both", expand=True)
    tk.Label(frame, text=APP_NAME, font=("Microsoft YaHei", 13, "bold"), fg="#0b3d91").pack(anchor="w")
    tk.Label(frame, text=f"v{APP_VERSION}　本地服务运行中", font=("Microsoft YaHei", 9), fg="#5b6b7f").pack(anchor="w", pady=(2, 10))
    tk.Label(frame, text=url, font=("Consolas", 11), fg="#1e6fd9").pack(anchor="w")
    tk.Label(
        frame,
        text="提示：关闭此窗口即退出程序；识别结果与简报保存在程序目录的 outputs 文件夹。",
        font=("Microsoft YaHei", 9),
        fg="#8a97a6",
        wraplength=420,
        justify="left",
    ).pack(anchor="w", pady=(8, 12))

    btns = tk.Frame(frame)
    btns.pack(anchor="e")

    def _quit() -> None:
        root.destroy()

    tk.Button(btns, text="打开界面", width=12, command=lambda: open_app_window(url)).pack(side="left", padx=(0, 8))
    tk.Button(btns, text="退出程序", width=12, command=_quit, bg="#0b3d91", fg="white").pack(side="left")

    root.protocol("WM_DELETE_WINDOW", _quit)
    root.mainloop()
    on_quit()


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=APP_NAME)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--headless", action="store_true", help="只起服务，不开任何窗口")
    ap.add_argument("--no-window", action="store_true", help="不开应用窗口")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--control", action="store_true", help="强制显示控制窗口")
    ap.add_argument("--version", action="store_true")
    ap.add_argument("--data-dir", help="结果、日志与任务库的可写根目录")
    ap.add_argument("--diagnose", metavar="JSON", help="检查本机连接并写入新的诊断JSON，不启动界面")
    commands = ap.add_mutually_exclusive_group()
    commands.add_argument("--batch", nargs=argparse.REMAINDER, help="批处理命令，后续参数与 scripts/run_batch.py 相同")
    commands.add_argument("--spectral", nargs=argparse.REMAINDER, help="光谱监测命令，后续参数与 scripts/run_spectral.py 相同")
    ap.add_argument("--evaluate", nargs=2, metavar=("PREDICTION", "REFERENCE"), help="离线评估两幅同网格0/1/255分类栅格")
    ap.add_argument("--evaluation-out", help="评估JSON新文件路径（不覆盖旧文件）")
    ap.add_argument("--terrain-profile", default="unspecified", help="精度评估的地貌标签")
    args = ap.parse_args(argv)
    if args.data_dir:
        os.environ["FLOOD_DATA_DIR"] = os.path.abspath(args.data_dir)

    if args.batch is not None or args.spectral is not None:
        setup_logging()
        bypass_proxy_for_localhost()
        if args.batch is not None:
            from scripts.run_batch import main as batch_main
            return batch_main(args.batch)
        from scripts.run_spectral import main as spectral_main
        return spectral_main(args.spectral)

    if args.diagnose:
        import json
        result = {"version": APP_VERSION, "frozen": bool(getattr(sys, "frozen", False)),
                  "bundle_root": bundle_root(), "data_root": user_root(),
                  "executable": sys.executable}
        try:
            result["loopback"] = check_loopback()
        except LoopbackError as exc:
            result["loopback"] = exc.as_dict()
        except OSError as exc:
            result["loopback"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        with open(args.diagnose, "x", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        return 0 if result["loopback"]["status"] == "ok" else 1

    if args.evaluate:
        import json
        from src.evaluation import evaluate_water
        if not args.evaluation_out:
            ap.error("--evaluate 需要 --evaluation-out 指定结果文件")
        result = evaluate_water(*args.evaluate, terrain_profile=args.terrain_profile)
        with open(args.evaluation_out, "x", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2, allow_nan=False)
        return 0

    if args.version:
        print(f"{APP_NAME} v{APP_VERSION}")
        return 0

    log_path = setup_logging()
    bypass_proxy_for_localhost()
    try:
        check_loopback()
    except OSError as exc:
        message = loopback_failure_message(exc, log_path)
        print(message, flush=True)
        if not args.headless and not args.no_window:
            try:
                import tkinter as tk
                from tkinter import messagebox
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror(APP_NAME, message, parent=root)
                root.destroy()
            except Exception:
                pass
        return 1
    os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(user_root(), "outputs", ".gradio"))
    started = start_startup_watchdog()

    from app.main import build_ui, _launch_kwargs
    from src.infer import available_weights

    weights = available_weights()
    print(f"检测到权重 {len(weights)} 个" + (f"：{os.path.basename(weights[0])}" if weights else "（将使用 NDWI 基线）"))

    demo = build_ui(baseline_only=args.baseline_only)
    try:
        demo.queue(default_concurrency_limit=1)
    except TypeError:
        demo.queue()

    # find_free_port 的"探测"与 Gradio 真正 bind 之间存在竞态窗口，
    # 中间被别的进程抢占时启动会直接失败。这里在失败后换端口重试。
    port = args.port
    url = ""
    last_exc: Optional[Exception] = None
    for attempt in range(5):
        port = find_free_port(args.host, args.port + attempt)
        url = f"http://{args.host}:{port}"
        try:
            demo.launch(
                server_name=args.host,
                server_port=port,
                share=False,
                show_error=True,
                inbrowser=False,
                quiet=True,
                prevent_thread_lock=True,
                **_launch_kwargs(),
            )
            last_exc = None
            break
        except OSError as exc:
            last_exc = exc
            print(f"[warn] 端口 {port} 启动失败（{exc}），换下一个端口重试")

    if last_exc is not None:
        raise last_exc
    started.set()
    print("=" * 70)
    print(f"{APP_NAME}  v{APP_VERSION}")
    print(f"资源目录：{bundle_root()}")
    print(f"工作目录：{user_root()}")
    print(f"权重目录：{weights_dirs()}")
    print(f"访问地址：{url}")
    if log_path:
        print(f"运行日志：{log_path}")
    print("=" * 70)
    print(f"✅ 服务已启动：{url}")

    stopped = threading.Event()

    def _shutdown() -> None:
        if stopped.is_set():
            return
        stopped.set()
        print("正在退出…")
        try:
            demo.close()
        except Exception:
            pass
        os._exit(0)  # 打包后必须硬退出，否则 uvicorn 线程会挂住进程

    if args.headless:
        print("（headless 模式：按 Ctrl+C 退出）")
        try:
            while not stopped.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            _shutdown()
        return 0

    if not args.no_window:
        mode = open_app_window(url)
        print(f"已打开界面（{mode}）")

    show_control = args.control or getattr(sys, "frozen", False)
    if show_control:
        control_window(url, _shutdown)
    else:
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            _shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
