"""Acceptance tests for PhysCrop-Risk report v3 kernel (Phase 2)."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from physcrop_agent_v2_core import (  # noqa: E402
    deterministic_report,
    merge_llm_narrative,
    read_json,
    validate_report,
)
from physcrop_agent_v2_risk_report import (  # noqa: E402
    EVIDENCE_DIR,
    REPORT_DIR,
    deterministic_risk_report,
    generate_risk_deterministic_reports,
    load_risk_knowledge,
    validate_risk_report,
)


class PhysCropRiskReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.knowledge = load_risk_knowledge()
        cls.report_paths = sorted(REPORT_DIR.glob("*.json"))
        if len(cls.report_paths) != 322:
            generate_risk_deterministic_reports()
            cls.report_paths = sorted(REPORT_DIR.glob("*.json"))
        cls.evidence_paths = sorted(
            path for path in EVIDENCE_DIR.glob("*.json") if path.stem != "risk_manifest"
        )

    def test_all_322_reports_generated_and_valid(self) -> None:
        self.assertEqual(322, len(self.report_paths))
        for path in self.report_paths:
            report = read_json(path)
            evidence = read_json(EVIDENCE_DIR / path.name)
            result = validate_report(report, evidence, self.knowledge)
            self.assertTrue(result["passes"], f"{path.name}: {result['errors']}")

    def test_report_v3_structure_and_policy_namespace(self) -> None:
        report = read_json(self.report_paths[0])
        self.assertEqual("physcrop_event_report_v3", report["schema_version"])
        self.assertEqual("post_audit_clean_null", report["risk"]["policy"])
        self.assertTrue(report["risk"]["not_a_diagnosis"])
        self.assertEqual([], report["actions"]["automatically_executed"])
        for item in report["professional_analysis"]["evidence_items"]:
            self.assertTrue(item["field_path"])
        self.assertIn("trajectory_assessment", report["professional_analysis"])
        self.assertIn("spatial_assessment", report["professional_analysis"])

    def test_prospective_reports_never_leak_outcomes(self) -> None:
        forbidden = ("target", "true_low", "covered_90", "covered_95", "harvest", "lead_days")
        for path in self.report_paths:
            text = path.read_text(encoding="utf-8")
            for key in forbidden:
                self.assertNotIn(f'"{key}"', text, f"{path.name} leaked {key}")

    def test_core_dispatch_routes_risk_schema(self) -> None:
        evidence = read_json(self.evidence_paths[0])
        report = deterministic_report(evidence, self.knowledge)
        self.assertEqual("physcrop_event_report_v3", report["schema_version"])
        validation = validate_report(report, evidence, self.knowledge)
        self.assertTrue(validation["passes"], validation["errors"])

    def test_tampered_prediction_fails(self) -> None:
        evidence = read_json(self.evidence_paths[0])
        report = deterministic_risk_report(evidence, self.knowledge)
        report["professional_analysis"]["evidence_items"][0]["value"] = 999.0
        result = validate_risk_report(report, evidence, self.knowledge)
        self.assertFalse(result["passes"])

    def test_pvalue_as_probability_fails(self) -> None:
        evidence = read_json(self.evidence_paths[0])
        report = deterministic_risk_report(evidence, self.knowledge)
        report["professional_analysis"]["narrative"] += "该田块低产概率为 30%。"
        result = validate_risk_report(report, evidence, self.knowledge)
        self.assertFalse(result["passes"])

    def test_spatial_as_cause_fails(self) -> None:
        # Frozen audit fields have no spatial maps (spatial maps belong to the
        # 316-field development challenge). Synthesize a spatial evidence block
        # to prove the validator rejects causal interpretation.
        evidence = copy.deepcopy(read_json(self.evidence_paths[0]))
        evidence["spatial_evidence"] = {
            "available": True,
            "map_asset_id": "maps/demo.npz",
            "pixels": 1000,
            "rmse_spatial_t_ha": 1.5,
            "r2_spatial": 0.2,
            "within_field_rho": 0.4,
            "bottom20_auroc": 0.7,
            "interpretation_boundary": "relative within-field evidence; not a causal or disease map",
        }
        report = deterministic_risk_report(evidence, self.knowledge)
        report["professional_analysis"]["spatial_assessment"]["narrative"] = "该图显示田块内病害病因，由严重缺水造成。"
        result = validate_risk_report(report, evidence, self.knowledge)
        self.assertFalse(result["passes"])

    def test_llm_cannot_modify_rule_owned_fields(self) -> None:
        report = read_json(self.report_paths[0])
        candidate = copy.deepcopy(report)
        candidate["event_key"] = "forged"
        candidate["risk"]["level"] = "action"
        candidate["actions"]["low_risk_now"] = ["立即使用未知药剂"]
        candidate["professional_analysis"]["evidence_items"][0]["interpretation"] = "伪造解释。"
        candidate["farmer_summary"]["headline"] = "这是经过语言优化的待核验说明。"
        candidate["professional_analysis"]["spatial_assessment"]["narrative_addition"] = "这里伪造一个 P99 的新数值。"
        merged = merge_llm_narrative(report, candidate)
        self.assertEqual(report["event_key"], merged["event_key"])
        self.assertEqual(report["risk"], merged["risk"])
        self.assertEqual(report["actions"], merged["actions"])
        self.assertEqual(report["professional_analysis"]["evidence_items"], merged["professional_analysis"]["evidence_items"])
        self.assertEqual("这是经过语言优化的待核验说明。", merged["farmer_summary"]["headline"])


if __name__ == "__main__":
    unittest.main()
