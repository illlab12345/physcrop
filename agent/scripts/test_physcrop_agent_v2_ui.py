"""Static UI serving smoke tests for app_v3 (Phase 4)."""

from __future__ import annotations

import sys
import threading
import unittest
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_physcrop_agent_v2_server import AgentHandler, V2_APP_DIR  # noqa: E402


class UiSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.assertTrue(V2_APP_DIR.is_dir(), "app_v3 must exist")
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            partial(AgentHandler, directory=str(V2_APP_DIR), demo=True, users={}),
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def fetch(self, path: str) -> tuple[int, str, str]:
        with urllib.request.urlopen(f"{self.base}{path}", timeout=10) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read().decode("utf-8")

    def test_v3_index_served_at_root(self) -> None:
        status, content_type, body = self.fetch("/")
        self.assertEqual(200, status)
        self.assertIn("text/html", content_type)
        self.assertIn("丰谷智警", body)

    def test_v3_assets_served(self) -> None:
        for path, marker in (("/app.js", '"use strict"'), ("/styles.css", "--brand: #1e7a50")):
            status, content_type, body = self.fetch(path)
            self.assertEqual(200, status)
            self.assertIn(marker, body)

    def test_dashboard_api_used_by_ui(self) -> None:
        status, _, body = self.fetch("/api/v2/dashboard/summary")
        self.assertEqual(200, status)
        self.assertIn("total_fields", body)


if __name__ == "__main__":
    unittest.main()
