# -*- coding: utf-8 -*-
"""端到端 API 测试：动态读取选项值，避免文本不一致"""
import sys, io, json, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from gradio_client import Client

c = Client("http://127.0.0.1:7860/")
info = c.view_api(return_format="dict")
params = info["named_endpoints"]["/run_single"]["parameters"]
mode_choices = params[3]["type"].get("enum") or params[3].get("choices")
weights_choices = params[8]["type"].get("enum") or params[8].get("choices")
print("mode choices:", mode_choices)
print("weights choices:", weights_choices)
mode = mode_choices[0]
weights = weights_choices[0]

print("\n=== 测试1：单景识别（demo01）===")
r = c.predict(None, None, "demo01", mode, 10.0, 120, 0.5, False, weights,
              api_name="/run_single")
print("overlay:", r[1])
print("summary:", (r[3] or "")[:200])
print("zip:", r[4])
print("log tail:", (r[5] or "")[-200:])

print("\n=== 测试2：灾前/灾后对比（demo02）===")
r2 = c.predict(None, None, "demo02", mode, 10.0, 120, 0.5, weights,
               api_name="/run_compare")
print("change_img:", r2[1])
print("summary:", (r2[3] or "")[:200])
print("zip:", r2[4])

ok = bool(r[1]) and bool(r2[1])
print("\n" + ("全部通过 ✓" if ok else "存在失败项 ✗"))
sys.exit(0 if ok else 1)
