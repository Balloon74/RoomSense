import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from roomsense.replay import main


class ReplayTests(unittest.TestCase):
    def test_cli_prints_session_records_in_chronological_order_without_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            lines = [
                {"schema_version": 1, "timestamp": 1.0, "kind": "session_start", "payload": {}},
                {"schema_version": 1, "timestamp": 2.0, "kind": "event", "payload": {"name": "SWIPE_RIGHT"}},
            ]
            path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = main([str(path)])

        self.assertEqual(result, 0)
        self.assertLess(output.getvalue().index("1.000"), output.getvalue().index("2.000"))
        self.assertIn("SWIPE_RIGHT", output.getvalue())

    def test_cli_reports_invalid_or_missing_sessions(self):
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            result = main(["/definitely/missing/roomsense-session.jsonl"])
        self.assertEqual(result, 2)
        self.assertIn("could not read session", error.getvalue())


if __name__ == "__main__":
    unittest.main()
