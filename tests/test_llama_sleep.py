"""llama-server --sleep-idle-seconds: polling must not hold off or undo the model's idle unload (#141)."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from gputaskman.llm import ServerInfo, ServerProber, build_view, probe_server, sleep_idle_enabled
from gputaskman.procinfo import ProcDetail
from gputaskman.report import _server_dict

BASE = "http://127.0.0.1:8080"
METRICS = "llamacpp:requests_processing 1\nllamacpp:requests_deferred 0\nllamacpp:kv_cache_usage_ratio 0.25\n"
SLOTS = [{"id": 0, "is_processing": True}, {"id": 1, "is_processing": False}]


def props(**extra) -> dict:
    return {"default_generation_settings": {"n_ctx": 4096}, "model_path": "/m/x.gguf", "total_slots": 2, **extra}


def probe(props_body: dict, **kwargs):
    responses = {BASE + "/props": props_body, BASE + "/v1/models": {"data": [{"id": "x"}]},
                 BASE + "/slots": SLOTS, BASE + "/metrics": METRICS}
    with patch("gputaskman.llm._get", side_effect=lambda url, **kw: responses[url]) as get:
        server = probe_server(8080, **kwargs)
    return server, [call.args[0].removeprefix(BASE) for call in get.call_args_list]


class SleepFlagTests(unittest.TestCase):
    def test_flag_values(self):
        cases = {
            (): False,
            ("--sleep-idle-seconds", "300"): True,
            ("--sleep-idle-seconds=0",): True,
            ("--sleep-idle-seconds", "-1"): False,
            ("--sleep-idle-seconds", "soon"): False,
            ("--sleep-idle-seconds",): False,  # no value
            ("--sleep-idle-seconds", "300", "--sleep-idle-seconds", "-1"): False,  # the last one wins
            ("--sleep-idle-seconds=-1", "--sleep-idle-seconds", "60"): True,
        }
        for args, expected in cases.items():
            with self.subTest(args=args):
                self.assertIs(sleep_idle_enabled(["llama-server", "-m", "x.gguf", *args]), expected)


class ProbeTests(unittest.TestCase):
    def test_without_sleep_every_endpoint_is_polled(self):
        server, paths = probe(props(is_sleeping=False, build_info="b12633-abc"))
        self.assertEqual(paths, ["/props", "/v1/models", "/slots", "/metrics"])
        self.assertIs(server.sleeping, False)
        self.assertEqual((server.slots_busy, server.slots_total, server.requests_running), (1, 2, 1))
        self.assertEqual(server.skipped, ())

    def test_sleeping_server_is_not_woken(self):
        for sleep_idle in (False, True):
            with self.subTest(sleep_idle=sleep_idle):
                server, paths = probe(props(is_sleeping=True, build_info="b12633-abc"), sleep_idle=sleep_idle)
                self.assertEqual(paths, ["/props", "/v1/models"])
                self.assertIs(server.sleeping, True)
                self.assertEqual(server.slots_total, 2)  # from /props, which never counts as activity
                self.assertIsNone(server.slots_busy)
                self.assertIsNone(server.requests_running)
                self.assertIsNone(server.kv_cache_usage)

    def test_sleep_enabled_skips_slots_and_keeps_metrics_on_new_builds(self):
        for build in ("b10519-abc", "b12633-abc"):
            with self.subTest(build=build):
                server, paths = probe(props(is_sleeping=False, build_info=build), sleep_idle=True)
                self.assertEqual(paths, ["/props", "/v1/models", "/metrics"])
                self.assertEqual(server.skipped, ("/slots",))
                self.assertIsNone(server.slots_busy)
                self.assertEqual((server.requests_running, server.requests_waiting), (1, 0))
                self.assertEqual(server.kv_cache_usage, 0.25)

    def test_sleep_enabled_skips_metrics_on_old_or_unknown_builds(self):
        for extra in ({"build_info": "b10518-abc"}, {"build_info": "unknown"}, {}):
            with self.subTest(extra=extra):
                server, paths = probe(props(**extra), sleep_idle=True)
                self.assertEqual(paths, ["/props", "/v1/models"])
                self.assertEqual(server.skipped, ("/slots", "/metrics"))
                self.assertIsNone(server.requests_running)

    def test_only_a_boolean_is_sleeping_counts(self):
        for value in ("true", 1, None):
            with self.subTest(value=value):
                server, paths = probe(props(is_sleeping=value))
                self.assertIsNone(server.sleeping)
                self.assertIn("/slots", paths)

    def test_prober_passes_the_flag(self):
        with patch("gputaskman.llm.probe_server", return_value=None) as p:
            ServerProber().get([8080], sleep_idle=True)
        self.assertIs(p.call_args.kwargs["sleep_idle"], True)


class ReportTests(unittest.TestCase):
    def test_json_has_sleeping_but_not_the_skip_list(self):
        server, _ = probe(props(is_sleeping=True))
        data = _server_dict(server)
        self.assertIs(data["sleeping"], True)
        self.assertNotIn("skipped", data)
        self.assertIsNone(_server_dict(ServerInfo("ollama", BASE))["sleeping"])  # unknown, not False

    def test_notes_say_why_readings_are_missing(self):
        def notes(server):
            return [n for n in build_view(ProcDetail(1, "llama-server"), 0, 0, server).notes if "sleep" in n]

        asleep, _ = probe(props(is_sleeping=True))
        self.assertIn("reports it is sleeping", notes(asleep)[0])
        new, _ = probe(props(is_sleeping=False, build_info="b12633-abc"), sleep_idle=True)
        self.assertIn("/slots is not polled", notes(new)[0])
        self.assertNotIn("/metrics is skipped", notes(new)[0])
        old, _ = probe(props(), sleep_idle=True)
        self.assertIn("/metrics is skipped", notes(old)[0])
        plain, _ = probe(props(is_sleeping=False))
        self.assertEqual(notes(plain), [])

    def test_sleeping_model_path_is_not_residency_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.gguf")
            with open(path, "wb") as f:
                f.write(b"GGUF")
            d = ProcDetail(1, "llama-server")
            with patch("gputaskman.llm.read_gguf", return_value=None):
                awake = build_view(d, 0, 0, ServerInfo("llama-server", BASE, model_path=path, sleeping=False))
                asleep = build_view(d, 0, 0, ServerInfo("llama-server", BASE, model_path=path, sleeping=True))
            self.assertTrue(any("api" in f.sources for f in awake.files + awake.other_files))
            self.assertFalse(any("api" in f.sources for f in asleep.files + asleep.other_files))


if __name__ == "__main__":
    unittest.main()
