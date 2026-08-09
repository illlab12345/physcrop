# PhysCrop-Risk LLM Report Agent

The agent turns the frozen PhysCrop-Risk results into a runnable, auditable
low-yield alerting workspace: alerts → evidence → constrained LLM polish →
automatic validation → human review → audit trail → PDF report.

## Structure

```text
agent/
  scripts/                  # adapter, report kernel, PDF, server, demo tools
  baseline_report/wacv/report_agent_v2/
    app_v3/                 # Chinese company-style frontend
    knowledge/              # governed risk knowledge base
    risk_evidence/          # 322-field evidence (prospective + retrospective)
    risk_reports/           # 322 Chinese reports + generation logs
  results/yieldsat_risk_v1/ # authoritative CSVs/JSONs used at runtime
  tests/                    # acceptance tests
  start_physcrop_demo.ps1   # one-click Windows demo launcher
```

## Install

```bash
pip install -r ../requirements.txt
```

## Run

```bash
cd agent
python scripts/prepare_company_demo.py          # idempotent data preparation
python scripts/run_physcrop_agent_v2_server.py --demo
```

Open `http://127.0.0.1:8765` (hard refresh with Ctrl+F5 if needed). The demo
launcher (`start_physcrop_demo.ps1`) prepares data, starts the server, and
opens the browser.

## Using the LLM polish

Set the provider key in the same terminal before starting the server:

```bash
export DEEPSEEK_API_KEY="sk-..."     # PowerShell: $env:DEEPSEEK_API_KEY="sk-..."
```

In the UI: open a field → 中文报告 → choose 深度求索 (or OpenAI/Anthropic) →
生成报告. The LLM may only rewrite narrative slots; all numbers, p-values,
alerts, and action lists stay rule-locked, and any failed validation falls back
to the deterministic report.

## Features

- 10-page Chinese workspace (overview, alerts, report center, fields, value
  calculator, tasks, knowledge, audit, retrospective, settings)
- Three-look evidence charts and decision tables
- Rule vs LLM polish comparison view
- Real-time generation stages + generation-log audit
- Report center with Markdown/PDF export (reportlab)
- Observation snapshots, human review, report version history
- Retrospective audit isolated from prospective alerts

## Tests

```bash
cd agent/scripts
python test_physcrop_agent_v2_risk.py
python test_physcrop_agent_v2_risk_report.py
python test_physcrop_agent_v2_api.py
python test_physcrop_agent_v2_ui.py
```

The API test spins up a real HTTP server on an ephemeral port; the UI test
checks static serving. Legacy demo data is intentionally not shipped, so the
legacy v1 demo test is not included.

## Boundaries

- The agent replays frozen audit results; it is not a real-time satellite/weather
  pipeline.
- It never trains or retunes the frozen predictor, calibration, or decision rules.
- Action alerts are conservative, require human review, and never trigger
  automatic agronomic intervention.
