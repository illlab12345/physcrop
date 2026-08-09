"""End-to-end LLM check for the PhysCrop-Risk report v3 pipeline.

Usage: python scripts/check_llm_connection.py [deepseek|openai|anthropic]
The provider API key must be present in the process environment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from physcrop_agent_v2_core import merge_llm_narrative, read_json  # noqa: E402
from physcrop_agent_v2_risk import EVIDENCE_DIR  # noqa: E402
from physcrop_agent_v2_risk_report import (  # noqa: E402
    build_risk_llm_messages,
    deterministic_risk_report,
    load_risk_knowledge,
    validate_risk_report,
)
from run_physcrop_agent_v2_server import call_llm  # noqa: E402


def pick_action_field() -> Path:
    for path in sorted(EVIDENCE_DIR.glob("*.json")):
        if path.stem == "risk_manifest":
            continue
        evidence = read_json(path)
        if evidence["model_evidence"]["action"]:
            return path
    raise SystemExit("No action field found")


def main() -> None:
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    persist = "--persist" in sys.argv
    provider = args[0] if args else "deepseek"
    if provider not in {"deepseek", "openai", "anthropic"}:
        raise SystemExit("provider must be deepseek/openai/anthropic")
    evidence_path = pick_action_field()
    evidence = read_json(evidence_path)
    knowledge = load_risk_knowledge()
    draft = deterministic_risk_report(evidence, knowledge)
    messages = build_risk_llm_messages(evidence, knowledge, draft)
    print(json.dumps({
        "field_id": evidence["identity"]["field_id"],
        "provider": provider,
        "draft_valid": validate_risk_report(draft, evidence, knowledge)["passes"],
    }, ensure_ascii=False, indent=2))
    try:
        candidate, model = call_llm(provider, messages)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "llm_failed_fallback", "error": str(exc)[:500]},
                         ensure_ascii=False, indent=2))
        return
    merged = merge_llm_narrative(draft, candidate)
    validation = validate_risk_report(merged, evidence, knowledge)
    persisted_path = None
    if persist and validation["passes"]:
        from physcrop_agent_v2_core import init_db
        from physcrop_agent_v2_risk_report import persist_risk_report
        connection = init_db()
        try:
            persisted_path = str(persist_risk_report(
                connection, merged, validation, "llm_guarded", provider, model
            ))
        finally:
            connection.close()
    result = {
        "status": "llm_guarded" if validation["passes"] else "validation_failed_fallback",
        "model": model,
        "validation_passes": validation["passes"],
        "validation_errors": validation["errors"][:5],
        "persisted_path": persisted_path,
        "headline": merged["farmer_summary"]["headline"][:120],
        "conclusion": merged["professional_analysis"]["conclusion"][:160],
        "narrative_has_addition": len(merged["professional_analysis"]["narrative"]) >
            len(draft["professional_analysis"]["narrative"]),
        "risk_level_unchanged": merged["risk"] == draft["risk"],
        "actions_unchanged": merged["actions"] == draft["actions"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
