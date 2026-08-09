"""Safety scan for the company demo: outcome leakage + forbidden phrasing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "baseline_report" / "wacv" / "report_agent_v2"
OUTCOME_KEYS = ('"target"', '"true_low"', '"covered_90"', '"covered_95"', '"harvest"', '"lead_days"')
FORBIDDEN_PHRASES = (
    "低产概率为", "低产概率是", "低产概率约", "已确诊", "确诊为",
    "保证增产", "自动灌溉", "自动施肥", "自动施药", "自动喷药",
    "由严重缺水造成", "明确属于涝害", "亩用",
)


def scan() -> list[tuple[str, str, str]]:
    violations: list[tuple[str, str, str]] = []
    paths = list((AGENT / "risk_evidence").glob("*.json")) + list((AGENT / "risk_reports").glob("*.json"))
    for path in paths:
        if path.stem == "risk_manifest":
            continue
        text = path.read_text(encoding="utf-8")
        for key in OUTCOME_KEYS:
            if key in text:
                violations.append((path.name, "outcome", key))
        for phrase in FORBIDDEN_PHRASES:
            if phrase in text:
                violations.append((path.name, "phrase", phrase))
    return violations


def main() -> None:
    violations = scan()
    print(f"扫描文件：{len(list((AGENT/'risk_evidence').glob('*.json'))) - 1 + len(list((AGENT/'risk_reports').glob('*.json')))}")
    print(f"违规：{len(violations)}")
    for name, kind, value in violations[:30]:
        print(f"  {kind}: {name}: {value}")
    if violations:
        raise SystemExit(1)
    print("通过：前瞻证据与报告中无 outcome 泄漏、无越界表达。")


if __name__ == "__main__":
    main()
