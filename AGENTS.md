# 开发协作与遥感验收

- GPT-6 Sol 负责需求理解、总体规划、困难问题、任务协调、集成审查、最终验证和用户沟通。
- 边界清楚且委派有收益的低难度工作，优先用本地 DeepSeek Harness（`deepseek_harness.delegate_task`；不可用时 `dsh --profile headless`）交给 DeepSeek V4.1 Flash。委派要限定工作目录、文件和验收条件；小任务可直接完成。
- 避免并行编辑同一文件。DeepSeek 的结果须由 GPT-6 Sol 审查和验证；不向委派任务提供凭据或无关私有数据。
- 不使用 AGY/Antigravity 或 Gemini 做委派，除非用户明确更改偏好。
- 遥感功能的验收须检查真实波段、投影与配准、逐波段 NoData、云/阴影质量、共同有效区、面积口径、日期和输入溯源。合成样本只验证流程，不能作为真实灾害精度结论。
