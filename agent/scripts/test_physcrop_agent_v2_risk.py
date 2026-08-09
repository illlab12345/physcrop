"""Acceptance tests for the YieldSAT / PhysCrop-Risk adapter (Phase 1)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from physcrop_agent_v2_core import canonical_hash, init_db, read_json  # noqa: E402
from physcrop_agent_v2_risk import (  # noqa: E402
    EVIDENCE_DIR,
    INPUT_HASH_PATH,
    RETRO_DIR,
    build_all_evidence,
    read_csv,
    seed_risk_events,
    verify_inputs,
    write_json,
)


class PhysCropRiskAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_all_evidence("retrospective_audit")
        cls.seed = seed_risk_events()
        cls.prospective_files = sorted(
            path for path in EVIDENCE_DIR.glob("*.json") if path.stem != "risk_manifest"
        )
        cls.retrospective_files = sorted(RETRO_DIR.glob("*.json"))

    def test_manifest_322_fields(self) -> None:
        self.assertEqual(322, self.manifest["field_count"])
        self.assertEqual(322, len(self.prospective_files))
        self.assertEqual(322, len(self.retrospective_files))

    def test_field_ids_hash_matches_freeze_manifest(self) -> None:
        freeze = read_json(
            Path(__file__).resolve().parents[1]
            / "results/yieldsat_risk_v1/final_audit/freeze_manifest.json"
        )
        self.assertEqual(freeze["field_ids_sha256"], self.manifest["field_ids_sha256"])

    def test_look_row_count_932(self) -> None:
        self.assertEqual(932, self.manifest["look_row_count"])

    def test_decision_counts_match_authoritative_reports(self) -> None:
        counts = self.manifest["decision_counts"]
        self.assertEqual(38, counts["watch_original"], "original frozen watch total")
        self.assertEqual(13, counts["action_original"], "original frozen action total")
        self.assertEqual(35, counts["watch_clean"], "clean-null watch total")
        self.assertEqual(12, counts["action_clean"], "clean-null action total")

    def test_prospective_outcome_leakage_is_zero(self) -> None:
        forbidden = ("target", "true_low", "covered_90", "covered_95", "harvest", "lead_days")
        for path in self.prospective_files:
            text = path.read_text(encoding="utf-8")
            for key in forbidden:
                self.assertNotIn(f'"{key}"', text, f"{path.name} leaked {key}")

    def test_evidence_hash_reproducible(self) -> None:
        for path in self.prospective_files:
            evidence = read_json(path)
            expected = evidence.pop("evidence_hash")
            self.assertEqual(expected, canonical_hash(evidence), path.name)

    def test_retrospective_block_sealed_only_in_retrospective(self) -> None:
        sample = read_json(self.retrospective_files[0])
        self.assertTrue(sample["retrospective_block"]["sealed_for_audit"])
        self.assertIn("target_t_ha", sample["retrospective_block"])
        self.assertNotIn("retrospective_block", read_json(self.prospective_files[0]))

    def test_input_manifest_matches_current_files(self) -> None:
        verify_inputs(strict=True)
        self.assertTrue(INPUT_HASH_PATH.exists())

    def test_input_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.sha256"
            write_json(bad, {"results/yieldsat_risk_v1/final_audit/final_predictions.csv": "0" * 64})
            with self.assertRaises(RuntimeError):
                verify_inputs(strict=True, manifest_path=bad)

    def test_database_seeded(self) -> None:
        connection = init_db()
        try:
            risk_events = connection.execute(
                "SELECT COUNT(*) FROM events WHERE source_type='yieldsat_risk_v1'"
            ).fetchone()[0]
            fields = connection.execute("SELECT COUNT(*) FROM fields").fetchone()[0]
            farms = connection.execute("SELECT COUNT(*) FROM farms").fetchone()[0]
            self.assertEqual(322, risk_events)
            self.assertEqual(322, fields)
            self.assertGreaterEqual(farms, 1)
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
