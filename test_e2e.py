# -*- coding: utf-8 -*-
"""端到端 API 测试：动态读取选项值，避免文本不一致。

改自原脚本，修掉两类"假成功"：
  1. 原判定只看两张叠加图路径非空（r[1] / r2[1]），成果包 zip 导出彻底坏掉、
     流程日志里满是 Traceback 时，脚本仍会打印"全部通过 ✓"并 exit(0)。
     现在对 zip 做存在性 + 非零字节校验，并断言流程日志不含 Traceback。
  2. 原脚本没有任何超时：后端卡死或没起来时会永久挂起（CI 里表现为任务不结束）。
     现在先做带超时的探活，再用看门狗给整轮测试设硬上限。

服务地址与超时可用环境变量覆盖（地址仅允许本机环回）：
    set HUIYAN_E2E_URL=http://127.0.0.1:7860/
    set HUIYAN_E2E_TIMEOUT=900

说明：run_single / run_compare 目前并未接入 app/main.py 的 build_ui（界面只保留
全流程 run_full_pipeline）。端点缺失时本脚本会明确失败，而不是打印"通过"。
"""
import io
import os
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from gradio_client import Client

DEFAULT_BASE = "http://127.0.0.1:7860/"
TIMEOUT = float(os.environ.get("HUIYAN_E2E_TIMEOUT", "900"))

# 端到端测试只应访问本机服务：限制协议与主机，避免被环境变量指向内网/外部地址
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")

failures = []


def resolve_base() -> str:
    """校验并返回测试目标地址（仅允许 http/https + 本机环回主机）。"""
    raw = os.environ.get("HUIYAN_E2E_URL", DEFAULT_BASE)
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise SystemExit(f"仅支持 http/https，收到：{raw}")
    host = (parsed.hostname or "").lower()
    if host not in _LOOPBACK_HOSTS:
        raise SystemExit(f"端到端测试只允许连接本机服务，收到主机：{host or '(空)'}")
    return raw


BASE = resolve_base()


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"  ✗ {msg}")


def check(cond: bool, msg: str) -> bool:
    if cond:
        print(f"  ✓ {msg}")
    else:
        fail(msg)
    return bool(cond)


def check_zip(path: object, tag: str) -> bool:
    """成果包必须存在且非空——只看路径非空是原脚本的假成功来源。"""
    if not path or not os.path.isfile(str(path)):
        fail(f"{tag}：成果包不存在（{path!r}）")
        return False
    size = os.path.getsize(str(path))
    if size <= 0:
        fail(f"{tag}：成果包是空文件（{path}）")
        return False
    print(f"  ✓ {tag}：成果包 {os.path.basename(str(path))} {size // 1024} KB")
    return True


def check_no_traceback(log: object, tag: str) -> bool:
    if log and "Traceback" in str(log):
        fail(f"{tag}：流程日志含 Traceback")
        return False
    print(f"  ✓ {tag}：日志无异常堆栈")
    return True


def probe() -> bool:
    """带超时的探活：避免服务没起来时脚本永久挂起。"""
    try:
        with urllib.request.urlopen(BASE, timeout=10) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError) as exc:
        print(f"无法连接 {BASE}（{type(exc).__name__}: {exc}）")
        print("请先启动服务：python app/main.py   或   python app/desktop.py --headless")
        return False


def print_summary() -> int:
    print()
    if failures:
        print(f"存在失败项 ✗（共 {len(failures)} 项）")
        for item in failures:
            print(f"  · {item}")
        return 1
    print("全部通过 ✓")
    return 0


def run_tests() -> int:
    client = Client(BASE)
    info = client.view_api(return_format="dict")
    endpoints = info.get("named_endpoints", {}) or {}

    missing = [name for name in ("/run_single", "/run_compare") if name not in endpoints]
    if missing:
        fail(f"服务未注册端点 {missing}；当前可用端点：{sorted(endpoints)}")
        print("\n端点缺失说明这两个回调没有接入 build_ui，本测试无法继续。")
        return print_summary()

    params = endpoints["/run_single"]["parameters"]
    mode_choices = params[3]["type"].get("enum") or params[3].get("choices")
    weights_choices = params[8]["type"].get("enum") or params[8].get("choices")
    print("mode choices:", mode_choices)
    print("weights choices:", weights_choices)
    if not mode_choices or not weights_choices:
        fail("未能从 API 定义读取 mode / weights 选项")
        return print_summary()
    mode = mode_choices[0]
    weights = weights_choices[0]

    print("\n=== 测试1：单景识别（demo01）===")
    r = client.predict(None, None, "demo01", mode, 10.0, 120, 0.5, False, weights,
                       api_name="/run_single")
    check(bool(r[1]), "单景：叠加图已生成")
    check(bool((r[3] or "").strip()), "单景：结论文本非空")
    check_zip(r[4] if len(r) > 4 else None, "单景")
    check_no_traceback(r[5] if len(r) > 5 else None, "单景")

    print("\n=== 测试2：灾前/灾后对比（demo02）===")
    r2 = client.predict(None, None, "demo02", mode, 10.0, 120, 0.5, weights,
                        api_name="/run_compare")
    check(bool(r2[1]), "对比：变化检测图已生成")
    check(bool((r2[3] or "").strip()), "对比：结论文本非空")
    check_zip(r2[4] if len(r2) > 4 else None, "对比")

    return print_summary()


def watchdog() -> None:
    print(f"\n[超时] 整轮测试超过 {TIMEOUT:.0f} 秒，强制结束（原脚本会永久挂起）")
    os._exit(2)


def main() -> int:
    if not probe():
        return 1
    timer = threading.Timer(TIMEOUT, watchdog)
    timer.daemon = True
    timer.start()
    try:
        return run_tests()
    finally:
        timer.cancel()


if __name__ == "__main__":
    sys.exit(main())
