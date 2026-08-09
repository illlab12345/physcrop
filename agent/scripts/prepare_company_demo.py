"""One-command, idempotent preparation of the PhysCrop-Risk company demo.

Verifies frozen inputs, builds evidence, seeds the SQLite workspace, and
generates deterministic reports when needed. Safe to run repeatedly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from physcrop_agent_v2_risk import (  # noqa: E402
    EVIDENCE_DIR,
    MANIFEST_PATH,
    build_all_evidence,
    seed_risk_events,
    verify_inputs,
)
from physcrop_agent_v2_risk_report import REPORT_DIR, generate_risk_deterministic_reports  # noqa: E402


def evidence_ready() -> bool:
    if not MANIFEST_PATH.exists():
        return False
    count = sum(1 for path in EVIDENCE_DIR.glob("*.json") if path.stem != "risk_manifest")
    return count == 322


def reports_ready() -> bool:
    return sum(1 for _ in REPORT_DIR.glob("*.json")) == 322


def main() -> None:
    print("== PhysCrop-Risk 公司演示准备 ==")
    print("[1/4] 校验冻结输入哈希 …")
    verify_inputs(strict=True)
    print("     输入文件哈希校验通过")
    print("[2/4] 构建 322 田块证据 …")
    if evidence_ready():
        print("     证据已存在，跳过")
    else:
        manifest = build_all_evidence("retrospective_audit")
        print(f"     已构建：{manifest['field_count']} 田块 / {manifest['look_row_count']} look 行")
    print("[3/4] 种子 SQLite 工作台 …")
    seeded = seed_risk_events()
    print(f"     已就绪：{seeded['field_count']} 田块 / {seeded['farm_count']} 农场")
    print("[4/4] 生成确定性报告 …")
    from physcrop_agent_v2_core import init_db
    connection = init_db()
    try:
        seeded_report_rows = connection.execute(
            """SELECT COUNT(*) FROM reports WHERE event_key IN (
                   SELECT event_key FROM events WHERE source_type='yieldsat_risk_v1')"""
        ).fetchone()[0]
    finally:
        connection.close()
    if reports_ready() and seeded_report_rows >= 322:
        print("     322 份报告已存在，跳过")
    else:
        paths = generate_risk_deterministic_reports()
        print(f"     已生成 {len(paths)} 份报告并写入数据库")
    connection = init_db()
    try:
        action_fields = [row[0] for row in connection.execute(
            "SELECT event_key FROM events WHERE source_type='yieldsat_risk_v1' AND action_clean=1 "
            "ORDER BY event_key LIMIT 6"
        ).fetchall()]
        pending = connection.execute(
            "SELECT COUNT(*) FROM reports WHERE status='needs_review'"
        ).fetchone()[0]
    finally:
        connection.close()
    print()
    print("== 演示就绪 ==")
    print(f"示例 action 田块：{', '.join(action_fields)}")
    print(f"待审核报告：{pending} 份")
    print("启动命令：python scripts/run_physcrop_agent_v2_server.py --demo")
    print("浏览器访问：http://127.0.0.1:8765 （旧版界面在 /legacy/）")


if __name__ == "__main__":
    main()
