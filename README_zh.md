# PhysCrop-Risk 投稿代码包说明

本包包含同一项目下的两部分：

1. **`reproduction/`**：论文《PhysCrop-Risk: Turning Partial-Season Earth Observation into Calibrated Low-Yield Alerts》的复现代码与冻结权威结果。
2. **`agent/`**：基于 PhysCrop-Risk 结果构建的“大模型报告 Agent”系统——可运行、可审计的低产预警工作台，含受约束 LLM 润色、人工审核、报告版本与 PDF 导出。

旧版 PhysCrop-F-v5 / 安达受控实验的代码与数据**已刻意排除**，不混入本提交包。

## 目录结构

```text
submission_physcrop_risk/
  README.md / README_zh.md / LICENSE / requirements.txt / MANIFEST.sha256
  paper/                         # paper.md 与 table.md
  reproduction/
    scripts/                     # 论文复现脚本
    results/yieldsat_risk_v1/    # 冻结权威输出（含审计模型）
  agent/
    scripts/                     # Agent 后端（适配器、报告、PDF、服务端）
    baseline_report/wacv/report_agent_v2/
      app_v3/                    # 全中文公司级前端
      knowledge/                 # 治理化风险知识库
      risk_evidence/             # 322 田块前瞻+回顾证据
      risk_reports/              # 322 份中文报告与生成日志
    results/yieldsat_risk_v1/    # Agent 运行时依赖的权威 CSV/JSON
    tests/                       # Agent 验收测试
    start_physcrop_demo.ps1      # Windows 一键演示脚本
```

## 环境要求

- Python 3.10+（开发于 3.10，Anaconda 3.12 亦可）
- 核心依赖：`numpy`、`scipy`、`scikit-learn`、`joblib`、`tifffile`、`matplotlib`、`pillow`、`reportlab`
- 可选（实验 10 视觉基线）：`torch`；AgriFM 冻结编码器还需 `einops`、`mmengine`、`timm`

```bash
pip install -r requirements.txt
```

## 诚信与边界（使用前必读）

- 一次性冻结审计不可重跑改结果：`reproduction/results/yieldsat_risk_v1/final_audit/final_report.json` 为权威输出，状态 `FAIL_FINAL`（仅最差分组 R² 门禁未过），如实保留、绝不改写。
- `post_audit_p0p1/` 中的 conformal-only（clean-null）结果是**事后诚信修正/对照**，不是第二次预注册揭示。
- 审计为 field-disjoint 而非 farm-disjoint；独立农场前瞻验证仍是后续工作。
- Agent 只读冻结结果：不训练、不调参、不改预测/p 值/告警决策。

## 快速开始

复现部分见 `reproduction/README.md`；Agent 部分见 `agent/README.md`，或直接：

```bash
cd agent
python scripts/prepare_company_demo.py
python scripts/run_physcrop_agent_v2_server.py --demo
# 浏览器打开 http://127.0.0.1:8765
```

## 校验

`MANIFEST.sha256` 列出包内全部文件的 SHA-256 摘要；Agent 验收测试位于 `agent/tests/`。
