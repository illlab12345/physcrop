"""HTTP integration acceptance tests for API v2 (Phase 3)."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_physcrop_agent_v2_server import (  # noqa: E402
    AgentHandler,
    LEGACY_APP_DIR,
    V2_APP_DIR,
    configured_users,
)


class AgentApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app_dir = V2_APP_DIR if V2_APP_DIR.is_dir() else LEGACY_APP_DIR
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            partial(AgentHandler, directory=str(cls.app_dir), demo=True, users={}),
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, path: str, method: str = "GET", body: dict | None = None,
                headers: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            f"{self.base}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_health_and_dashboard(self) -> None:
        status, health = self.request("/api/v2/health")
        self.assertEqual(200, status)
        self.assertIn("PhysCrop-Risk", health["system"])
        status, dashboard = self.request("/api/v2/dashboard/summary")
        self.assertEqual(200, status)
        self.assertEqual(322, dashboard["portfolio"]["total_fields"])
        self.assertEqual(12, dashboard["portfolio"]["action_alerts"])
        self.assertEqual(23, dashboard["portfolio"]["watch_alerts"])
        self.assertEqual("FAIL_FINAL", dashboard["authoritative"]["audit_status"])

    def test_alerts_list_filters_and_pagination(self) -> None:
        status, payload = self.request("/api/v2/alerts?limit=10")
        self.assertEqual(200, status)
        self.assertEqual(322, payload["count"])
        self.assertEqual(10, len(payload["alerts"]))
        status, payload = self.request("/api/v2/alerts?level=action")
        self.assertEqual(200, status)
        self.assertEqual(12, len(payload["alerts"]))
        status, payload = self.request("/api/v2/alerts?country=Uruguay&crop=soybean")
        self.assertEqual(200, status)
        self.assertTrue(all(item["country"] == "Uruguay" and item["crop"] == "soybean"
                            for item in payload["alerts"]))

    def test_alert_detail_has_evidence_report_and_no_outcome(self) -> None:
        status, listing = self.request("/api/v2/alerts?level=action&limit=1")
        field_id = listing["alerts"][0]["field_id"]
        status, detail = self.request(f"/api/v2/alerts/{field_id}")
        self.assertEqual(200, status)
        self.assertEqual("physcrop_risk_alert_evidence_v1", detail["evidence"]["schema_version"])
        self.assertIsNotNone(detail["report"])
        self.assertEqual("physcrop_event_report_v3", detail["report"]["schema_version"])
        self.assertIn("rule", detail["compare"])
        self.assertIn("llm", detail["compare"])
        self.assertGreaterEqual(len(detail["report_versions"]), 1)
        text = json.dumps(detail["evidence"], ensure_ascii=False)
        for key in ("target", "true_low", "covered_90", "covered_95"):
            self.assertNotIn(f'"{key}"', text)

    def test_spatial_knowledge_audit_endpoints(self) -> None:
        status, payload = self.request("/api/v2/fields/spatial-summary")
        self.assertEqual(200, status)
        self.assertEqual(316, payload["count"])
        status, payload = self.request("/api/v2/knowledge")
        self.assertEqual(200, status)
        self.assertGreaterEqual(len(payload["entries"]), 5)
        status, payload = self.request("/api/v2/audit")
        self.assertEqual(200, status)
        self.assertGreaterEqual(len(payload["audit_log"]), 1)

    def test_retrospective_summary_sealed(self) -> None:
        status, payload = self.request("/api/v2/retrospective/summary")
        self.assertEqual(200, status)
        self.assertEqual("retrospective_audit", payload["mode"])
        self.assertEqual(932, len(payload["fields"]))
        self.assertIn("target", payload["fields"][0])

    def test_observation_snapshot_flow(self) -> None:
        status, listing = self.request("/api/v2/alerts?limit=1")
        field_id = listing["alerts"][0]["field_id"]
        _, before = self.request(f"/api/v2/alerts/{field_id}")
        before_count = len(before["snapshots"])
        status, created = self.request(
            f"/api/v2/alerts/{field_id}/observations",
            method="POST",
            body={"soil_moisture": "偏干", "standing_water": "未见",
                  "canopy_symptoms": "叶色略淡", "notes": "现场快照测试"},
        )
        self.assertEqual(201, status)
        self.assertTrue(created["snapshot"]["immutable"])
        status, detail = self.request(f"/api/v2/alerts/{field_id}")
        self.assertEqual(200, status)
        self.assertEqual(before_count + 1, len(detail["snapshots"]))

    def test_report_generate_job_and_idempotency(self) -> None:
        status, listing = self.request("/api/v2/alerts?limit=1")
        field_id = listing["alerts"][0]["field_id"]
        key = f"phase3-idempotency-{int(time.time() * 1000)}"
        status, job = self.request(
            "/api/v2/reports/generate",
            method="POST",
            body={"event_key": field_id, "provider": "local", "idempotency_key": key},
        )
        self.assertEqual(202, status)
        job_id = job["job_id"]
        result = None
        for _ in range(100):
            status, job = self.request(f"/api/v2/jobs/{job_id}")
            if job["status"] == "done":
                result = job["result"]
                break
            time.sleep(0.1)
        self.assertIsNotNone(result)
        self.assertTrue(result["validation"]["passes"], result["validation"]["errors"])
        self.assertEqual("physcrop_event_report_v3", result["report"]["schema_version"])
        status, cached = self.request(
            "/api/v2/reports/generate",
            method="POST",
            body={"event_key": field_id, "provider": "local", "idempotency_key": key},
        )
        self.assertEqual(200, status)
        self.assertTrue(cached["cached"])

    def test_markdown_export(self) -> None:
        status, listing = self.request("/api/v2/alerts?limit=1")
        field_id = listing["alerts"][0]["field_id"]
        request = urllib.request.Request(f"{self.base}/api/v2/export/report/{field_id}.md")
        with urllib.request.urlopen(request, timeout=15) as response:
            text = response.read().decode("utf-8")
        self.assertTrue(text.startswith("# PhysCrop-Risk 田块报告"))
        self.assertIn("证据哈希", text)

    def test_pdf_export(self) -> None:
        status, listing = self.request("/api/v2/alerts?limit=1")
        field_id = listing["alerts"][0]["field_id"]
        request = urllib.request.Request(f"{self.base}/api/v2/export/report/{field_id}.pdf")
        with urllib.request.urlopen(request, timeout=20) as response:
            content_type = response.headers.get("Content-Type", "")
            body = response.read()
        self.assertIn("application/pdf", content_type)
        self.assertTrue(body.startswith(b"%PDF"))
        self.assertGreater(len(body), 2000)

    def test_report_center_and_generation_logs(self) -> None:
        status, payload = self.request("/api/v2/reports?limit=20")
        self.assertEqual(200, status)
        self.assertGreater(payload["count"], 0)
        self.assertGreaterEqual(len(payload["reports"]), 1)
        self.assertIn("generator", payload["reports"][0])
        status, payload = self.request("/api/v2/reports?generator=llm_guarded")
        self.assertEqual(200, status)
        self.assertTrue(all(row["generator"] == "llm_guarded" for row in payload["reports"]))

        status, listing = self.request("/api/v2/alerts?limit=1")
        field_id = listing["alerts"][0]["field_id"]
        key = f"genlog-test-{int(time.time() * 1000)}"
        status, job = self.request(
            "/api/v2/reports/generate",
            method="POST",
            body={"event_key": field_id, "provider": "local", "idempotency_key": key},
        )
        self.assertEqual(202, status)
        job_id = job["job_id"]
        for _ in range(100):
            status, job = self.request(f"/api/v2/jobs/{job_id}")
            if job["status"] == "done":
                break
            time.sleep(0.1)
        self.assertEqual("done", job["status"])
        self.assertGreaterEqual(len(job.get("events", [])), 3)
        self.assertEqual("done", job.get("stage"))
        status, payload = self.request("/api/v2/generation-logs")
        self.assertEqual(200, status)
        self.assertGreaterEqual(payload["count"], 1)
        self.assertTrue(any(row["event_key"] == field_id for row in payload["logs"]))

    def test_resource_permission_denies_other_farm(self) -> None:
        os.environ["PHYSCROP_AGENT_USERS_JSON"] = json.dumps({
            "token-farmer-a": {
                "username": "farmer-a",
                "role": "farmer",
                "farms": ["Argentina_DUP1_farm11"],
            }
        })
        try:
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                partial(AgentHandler, directory=str(self.app_dir), demo=False,
                        users=configured_users()),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            headers = {"Authorization": "Bearer token-farmer-a"}
            try:
                request = urllib.request.Request(
                    f"{base}/api/v2/alerts", headers={**headers, "Content-Type": "application/json"}
                )
                with urllib.request.urlopen(request, timeout=15) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self.assertGreater(payload["count"], 0)
                self.assertTrue(all(item["farm_id"] == "Argentina_DUP1_farm11"
                                    for item in payload["alerts"]))
                request = urllib.request.Request(
                    f"{base}/api/v2/alerts/Brazil_DUP2_farm5_field143_corn_2020",
                    headers={**headers, "Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(request, timeout=15)
                self.assertEqual(403, ctx.exception.code)
            finally:
                server.shutdown()
                server.server_close()
        finally:
            os.environ.pop("PHYSCROP_AGENT_USERS_JSON", None)


if __name__ == "__main__":
    unittest.main()
