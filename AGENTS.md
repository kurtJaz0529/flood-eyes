# 开发协作与遥感验收

> 上一轮（2026-09-26）因 Codex 额度耗尽中断，收尾内容、已修 bug、未完成项与接手顺序见
> [交接说明_20260926](docs/交接说明_20260926.md)。恢复本仓库工作前先读它，避免重复劳动。

- GPT-6 Sol 负责需求理解、总体规划、困难问题、任务协调、集成审查、最终验证和用户沟通。
- 边界清楚且委派有收益的低难度工作，优先用 WorkBuddy MCP `workbuddy_run`，显式指定 `model="deepseek-v4.1-flash"`。MCP 未加载时使用 `python D:\agent\codex\workbuddy-bridge\workbuddy_bridge.py --task-file <UTF-8任务文件> --workspace <绝对目录>`。委派限定工作目录、文件、权限和验收条件；小任务直接完成。DeepSeek Harness 保留为另行选择的独立入口，不自动跨服务重试。
- WorkBuddy 超时或额度/网络失败不自动重复发起，先检查部分文件。GPT 负责算法、真实波段、投影配准、NoData、面积口径和最后验收；避免多个 GPT 子代理重复阅读消耗额度。
- 避免并行编辑同一文件。DeepSeek 的结果须由 GPT-6 Sol 审查和验证；不向委派任务提供凭据或无关私有数据。
- 不使用 AGY/Antigravity 或 Gemini 做委派，除非用户明确更改偏好。
- 遥感功能的验收须检查真实波段、投影与配准、逐波段 NoData、云/阴影质量、共同有效区、面积口径、日期和输入溯源。合成样本只验证流程，不能作为真实灾害精度结论。
