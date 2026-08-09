# PhysCrop-Risk Submission Package

This package contains two parts built on the same project:

1. **`reproduction/`** — the code and frozen outputs that reproduce the paper
   *PhysCrop-Risk: Turning Partial-Season Earth Observation into Calibrated Low-Yield Alerts*.
2. **`agent/`** — the LLM report agent system built on the PhysCrop-Risk results:
   a runnable, auditable low-yield alerting workspace with constrained LLM polish,
   human review, report versioning, and PDF export.

Legacy PhysCrop-F-v5 / Ando controlled-experiment code and data are intentionally
**not** included in this package.

## Package layout

```text
submission_physcrop_risk/
  README.md / README_zh.md / LICENSE / requirements.txt / MANIFEST.sha256
  paper/                         # paper.md and table.md
  reproduction/
    scripts/                     # PhysCrop-Risk reproduction scripts
    results/yieldsat_risk_v1/    # frozen authoritative outputs (incl. audit models)
  agent/
    scripts/                     # agent backend (adapter, report, PDF, server)
    baseline_report/wacv/report_agent_v2/
      app_v3/                    # Chinese company-style frontend
      knowledge/                 # governed risk knowledge base
      risk_evidence/             # 322-field prospective + retrospective evidence
      risk_reports/              # 322 generated Chinese reports + generation logs
    results/yieldsat_risk_v1/    # authoritative CSVs/JSONs used by the agent
    tests/                       # agent acceptance tests
    start_physcrop_demo.ps1      # one-click demo launcher (Windows)
```

## Requirements

- Python 3.10+ (developed on 3.10; Anaconda 3.12 also works)
- Core: `numpy`, `scipy`, `scikit-learn`, `joblib`, `tifffile`, `matplotlib`,
  `pillow`, `reportlab`
- Optional (Experiment 10 vision baselines): `torch`, and for the frozen AgriFM
  encoder `einops`, `mmengine`, `timm`

```bash
pip install -r requirements.txt
```

## Integrity notes (read before use)

- The one-shot final audit is frozen and **must not be re-run to alter results**:
  `reproduction/results/yieldsat_risk_v1/final_audit/final_report.json` is the
  authoritative output and its status is `FAIL_FINAL` (only the worst-group R2
  gate failed). This failure is reported honestly and never rewritten.
- The conformal-only (clean-null) results in `post_audit_p0p1/` are a
  **post-audit integrity correction/comparison**, not a second pre-registered reveal.
- The audit is field-disjoint but not farm-disjoint; deployment validation on
  unseen farms remains future work.
- The agent reads frozen outputs only. It never trains, tunes, or modifies the
  frozen predictions, p-values, or alert decisions.

## Quick start

Reproduction: see `reproduction/README.md`.

Agent: see `agent/README.md`, or run:

```bash
cd agent
python scripts/prepare_company_demo.py
python scripts/run_physcrop_agent_v2_server.py --demo
# open http://127.0.0.1:8765
```

## Verification

`MANIFEST.sha256` lists every file in this package with its SHA-256 digest.
Agent acceptance tests live in `agent/tests/`.
