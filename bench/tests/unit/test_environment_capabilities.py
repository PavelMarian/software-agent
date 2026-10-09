from __future__ import annotations

import json
import shutil
import sys
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from software_bench.core.models import TaskBundle
from software_bench.core.task_bundle import load_task_bundle
from software_bench.harness.environments.http import ConfiguredHttpTools
from software_bench.harness.environments import create_environment_session


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        size = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(size))
        payload = json.dumps({"accepted": body}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *args: object) -> None:
        return None


def test_configured_http_tool_records_request_and_response(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = ConfiguredHttpTools(
            {
                "http_tools": {
                    "base_url": f"http://127.0.0.1:{server.server_port}",
                    "tools": [
                        {
                            "name": "submit",
                            "parameters": {"type": "object"},
                            "method": "POST",
                            "path": "/submit",
                            "body_arguments": ["stage", "solution"],
                            "record_path": "submissions/{stage}.json",
                            "record_request": True,
                        }
                    ],
                }
            },
            write_text=lambda name, text: (tmp_path / name).parent.mkdir(
                parents=True, exist_ok=True
            )
            or (tmp_path / name).write_text(text, encoding="utf-8"),
        )
        result = provider.invoke(
            "submit", {"stage": "diagnosis", "solution": "root cause"}
        )
        recorded = json.loads(
            (tmp_path / "submissions/diagnosis.json").read_text(encoding="utf-8")
        )
        assert result.exit_code == 0
        assert recorded["request"]["solution"] == "root cause"
        assert recorded["response"]["accepted"]["stage"] == "diagnosis"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_managed_process_allocates_port_and_exposes_tools(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "fixtures/task_bundle"
    root = tmp_path / "bundle"
    shutil.copytree(fixture, root)
    server = root / "evaluator/server.py"
    server.parent.mkdir(exist_ok=True)
    server.write_text(
        """import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body=json.dumps({'ready': True}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args): pass
ThreadingHTTPServer(('127.0.0.1', int(os.environ['PORT'])), H).serve_forever()
""",
        encoding="utf-8",
    )
    loaded = load_task_bundle(root)
    config = {
        "managed_process": {
            "command": [sys.executable, "{bundle_root}/evaluator/server.py"],
            "readiness_url": "http://127.0.0.1:{port}/ready",
            "allocate_port": True,
            "environment": {"PORT": "{port}"},
        },
        "http_tools": {
            "base_url": "http://127.0.0.1:{port}",
            "tools": [
                {
                    "name": "status",
                    "parameters": {"type": "object"},
                    "method": "GET",
                    "path": "/ready",
                }
            ],
        },
    }
    environment = replace(
        loaded.environment,
        backend_hint="managed_process",
        workspace_source="bundle",
        seed_path="workspace",
        backend_config=config,
    )
    bundle = TaskBundle(
        loaded.root, loaded.task, loaded.evaluation, environment, loaded.provenance
    )

    session = create_environment_session(
        "managed_process", bundle, workspace=None, run_id="managed-test"
    )
    try:
        assert "status" in session.tool_declarations()
        assert json.loads(session.invoke_tool("status", {}).stdout) == {"ready": True}
    finally:
        session.close()
