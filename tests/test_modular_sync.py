import http.client
import base64
import json
import threading
import time
import unittest
from unittest import mock
from urllib.parse import parse_qs

import tempfile
import os
from datetime import datetime
from gemini_web2api.config import CONFIG, DEFAULT_CONFIG, load_config
from gemini_web2api.gemini import _build_payload, _build_headers, _get_url, DEFAULT_GEMINI_BL
from gemini_web2api.server import GeminiHandler, ThreadedServer
from gemini_web2api.tools import google_contents_to_prompt, messages_to_prompt
from gemini_web2api.audit import get_audit_log_path, record_audit_log
from gemini_web2api.rate_limiter import calculate_interval, RateLimiter, rate_limiter


def _decode_payload(payload):
    outer = json.loads(parse_qs(payload)["f.req"][0])
    return json.loads(outer[1])


def _decode_sse(body):
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        event_type = next(
            (line[len("event: "):] for line in lines if line.startswith("event: ")),
            None,
        )
        data = next(
            (line[len("data: "):] for line in lines if line.startswith("data: ")),
            None,
        )
        if event_type and data:
            events.append((event_type, json.loads(data)))
    return events


class PayloadPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.original_config = dict(CONFIG)

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def test_temporary_chats_default_to_disabled(self):
        self.assertIs(DEFAULT_CONFIG["temporary_chats"], False)

    def test_persistent_chat_payload(self):
        CONFIG["temporary_chats"] = False

        inner = _decode_payload(_build_payload("hello", 1, 4))

        self.assertEqual(inner[41], [2])
        self.assertIsNone(inner[45])

    def test_temporary_chat_payload(self):
        CONFIG["temporary_chats"] = True

        inner = _decode_payload(_build_payload("hello", 1, 4))

        self.assertEqual(inner[41], [1])
        self.assertEqual(inner[45], 1)

    def test_payload_includes_uploaded_image_refs(self):
        inner = _decode_payload(_build_payload("describe", 1, 4, ["/uploaded/image-ref"]))

        self.assertEqual(inner[0][0], "describe")
        self.assertEqual(inner[0][3], [[None, None, "/uploaded/image-ref"]])


class MessageParsingTests(unittest.TestCase):
    def test_messages_to_prompt_extracts_openai_image_url_data_url(self):
        image_data = base64.b64encode(b"fake png").decode()

        prompt, images = messages_to_prompt([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
            ],
        }])

        self.assertEqual(prompt, "Describe [Image attached]")
        self.assertEqual(images, [(b"fake png", "image/png")])

    def test_messages_to_prompt_extracts_responses_input_image_url(self):
        prompt, images = messages_to_prompt([{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe"},
                {"type": "input_image", "image_url": "https://example.com/image.png"},
            ],
        }])

        self.assertEqual(prompt, "Describe [Image attached]")
        self.assertEqual(images, [("https://example.com/image.png", "image/png")])

    def test_messages_to_prompt_ignores_malformed_image_data_url(self):
        prompt, images = messages_to_prompt([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%"}},
            ],
        }])

        self.assertEqual(prompt, "Describe")
        self.assertEqual(images, [])

    def test_google_contents_to_prompt_extracts_inline_image_data(self):
        image_data = base64.b64encode(b"fake png").decode()

        prompt, images = google_contents_to_prompt({
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": "Describe"},
                    {"inlineData": {"mimeType": "image/png", "data": image_data}},
                ],
            }],
        })

        self.assertEqual(prompt, "Describe\n[Image attached]")
        self.assertEqual(images, [(b"fake png", "image/png")])

    def test_google_contents_to_prompt_ignores_malformed_inline_image_data(self):
        prompt, images = google_contents_to_prompt({
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": "Describe"},
                    {"inlineData": {"mimeType": "image/png", "data": "%%%"}},
                ],
            }],
        })

        self.assertEqual(prompt, "Describe")
        self.assertEqual(images, [])


class StreamingEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadedServer(("127.0.0.1", 0), GeminiHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.original_config = dict(CONFIG)
        CONFIG["api_keys"] = []
        CONFIG["log_requests"] = False

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def post_json(self, path, payload):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(
            "POST",
            path,
            body=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = response.read().decode()
        headers = dict(response.getheaders())
        connection.close()
        return response.status, headers, body

    def post_chunked_json(self, path, payload):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(
            "POST",
            path,
            body=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            encode_chunked=True,
        )
        response = connection.getresponse()
        body = response.read().decode()
        headers = dict(response.getheaders())
        connection.close()
        return response.status, headers, body

    @mock.patch("gemini_web2api.server.generate_stream")
    def test_chat_stream_starts_with_assistant_role(self, generate_stream):
        generate_stream.return_value = iter(["hel", "lo"])

        status, headers, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        chunks = [
            json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: {")
        ]
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"content": "hel"})
        self.assertEqual(chunks[2]["choices"][0]["delta"], {"content": "lo"})
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    @mock.patch("gemini_web2api.server.generate", return_value="chunked ok")
    def test_chat_accepts_chunked_body(self, _generate):
        status, _, body = self.post_chunked_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "chunked ok")

    @mock.patch("gemini_web2api.server.upload_image", return_value="/uploaded/image-ref")
    @mock.patch("gemini_web2api.server.generate", return_value="looks good")
    def test_chat_accepts_openai_image_url_data_url(self, generate, upload_image):
        image_data = base64.b64encode(b"fake png").decode()

        status, _, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            },
                        },
                    ],
                }],
            },
        )

        self.assertEqual(status, 200)
        upload_image.assert_called_once_with(b"fake png", "image.png", "image/png")
        self.assertEqual(generate.call_args.args[3], ["/uploaded/image-ref"])
        self.assertIn("[Image attached]", generate.call_args.args[0])
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "looks good")

    @mock.patch("gemini_web2api.server.fetch_image_bytes", return_value=b"\xff\xd8\xffremote jpeg")
    @mock.patch("gemini_web2api.server.upload_image", return_value="/uploaded/remote-ref")
    @mock.patch("gemini_web2api.server.generate", return_value="remote ok")
    def test_responses_accepts_input_image_url(self, generate, upload_image, fetch_image_bytes):
        status, _, _ = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": [{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is shown?"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/image.jpg",
                        },
                    ],
                }],
            },
        )

        self.assertEqual(status, 200)
        fetch_image_bytes.assert_called_once_with("https://example.com/image.jpg")
        upload_image.assert_called_once_with(b"\xff\xd8\xffremote jpeg", "image.png", "image/jpeg")
        self.assertEqual(generate.call_args.args[3], ["/uploaded/remote-ref"])
        self.assertIn("[Image attached]", generate.call_args.args[0])

    @mock.patch("gemini_web2api.server.upload_image", return_value="/uploaded/image-ref")
    @mock.patch("gemini_web2api.server.generate", return_value="top-level image ok")
    def test_responses_accepts_top_level_input_image(self, generate, upload_image):
        image_data = base64.b64encode(b"fake png").decode()

        status, _, _ = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": [
                    {"type": "input_text", "text": "What is shown?"},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_data}",
                    },
                ],
            },
        )

        self.assertEqual(status, 200)
        upload_image.assert_called_once_with(b"fake png", "image.png", "image/png")
        self.assertEqual(generate.call_args.args[3], ["/uploaded/image-ref"])
        self.assertIn("What is shown?", generate.call_args.args[0])
        self.assertIn("[Image attached]", generate.call_args.args[0])

    @mock.patch("gemini_web2api.server.upload_image", side_effect=RuntimeError("upload denied"))
    def test_google_image_upload_failure_returns_502(self, _upload_image):
        image_data = base64.b64encode(b"fake png").decode()

        status, _, body = self.post_json(
            "/v1beta/models/gemini-3.6-flash:generateContent",
            {
                "contents": [{
                    "role": "user",
                    "parts": [{
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": image_data,
                        },
                    }],
                }],
            },
        )

        self.assertEqual(status, 502)
        self.assertIn("image upload failed: upload denied", json.loads(body)["error"]["message"])

    @mock.patch("gemini_web2api.server.generate_stream", return_value=iter(["streamed"]))
    def test_google_stream_generate_content_uses_sse(self, _generate_stream):
        status, headers, body = self.post_json(
            "/v1beta/models/gemini-3.6-flash:streamGenerateContent",
            {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": "Stream this"}],
                }],
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        self.assertIn('"text": "streamed"', body)

    @mock.patch("gemini_web2api.server.generate", return_value="hello")
    def test_responses_text_stream_has_complete_event_sequence(self, _generate):
        status, headers, body = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": "hello",
                "stream": True,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        events = _decode_sse(body)
        self.assertEqual(
            [event_type for event_type, _ in events],
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        self.assertEqual(
            [event["sequence_number"] for _, event in events],
            list(range(1, len(events) + 1)),
        )
        self.assertEqual(events[4][1]["delta"], "hello")
        self.assertEqual(events[-1][1]["response"]["status"], "completed")
        self.assertEqual(events[-1][1]["response"]["output"][0]["content"][0]["text"], "hello")

    @mock.patch("gemini_web2api.server.parse_tool_calls")
    @mock.patch("gemini_web2api.server.generate", return_value="tool output")
    def test_responses_function_call_stream_has_complete_event_sequence(
        self, _generate, parse_tool_calls
    ):
        parse_tool_calls.return_value = (
            "",
            [
                {
                    "id": "call_test",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Shanghai"}'},
                }
            ],
        )

        status, _, body = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": "weather",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    }
                ],
                "stream": True,
            },
        )

        self.assertEqual(status, 200)
        events = _decode_sse(body)
        self.assertEqual(
            [event_type for event_type, _ in events],
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        self.assertEqual(
            [event["sequence_number"] for _, event in events],
            list(range(1, len(events) + 1)),
        )
        self.assertEqual(events[2][1]["output_index"], 0)
        self.assertEqual(events[3][1]["delta"], '{"city":"Shanghai"}')
        self.assertEqual(events[4][1]["arguments"], '{"city":"Shanghai"}')
        self.assertEqual(events[-1][1]["response"]["output"][0]["name"], "get_weather")

    @mock.patch("gemini_web2api.server.generate", return_value="audit test response")
    def test_chat_audit_log_when_enabled(self, _generate):
        with tempfile.TemporaryDirectory() as temp_dir:
            CONFIG["audit_log"] = True
            CONFIG["audit_log_dir"] = temp_dir

            status, _, _ = self.post_json(
                "/v1/chat/completions",
                {
                    "model": "gemini-3.6-flash",
                    "messages": [{"role": "user", "content": "hello audit"}],
                },
            )
            self.assertEqual(status, 200)

            files = os.listdir(temp_dir)
            self.assertEqual(len(files), 1)
            with open(os.path.join(temp_dir, files[0]), "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]

            self.assertEqual(len(lines), 1)
            entry = lines[0]
            self.assertEqual(entry["model"], "gemini-3.6-flash")
            self.assertEqual(entry["client_ip"], "127.0.0.1")
            self.assertIn("timestamp", entry)
            self.assertEqual(entry["request"]["messages"][0]["content"], "hello audit")
            self.assertEqual(entry["response"]["choices"][0]["message"]["content"], "audit test response")

    @mock.patch("gemini_web2api.server.generate", return_value="audit test response")
    def test_audit_log_not_written_when_disabled(self, _generate):
        with tempfile.TemporaryDirectory() as temp_dir:
            CONFIG["audit_log"] = False
            CONFIG["audit_log_dir"] = temp_dir

            status, _, _ = self.post_json(
                "/v1/chat/completions",
                {
                    "model": "gemini-3.6-flash",
                    "messages": [{"role": "user", "content": "hello audit disabled"}],
                },
            )
            self.assertEqual(status, 200)
            self.assertEqual(len(os.listdir(temp_dir)), 0)

    @mock.patch("gemini_web2api.server.generate_stream")
    def test_chat_streaming_audit_log_records_full_response(self, generate_stream):
        with tempfile.TemporaryDirectory() as temp_dir:
            CONFIG["audit_log"] = True
            CONFIG["audit_log_dir"] = temp_dir
            generate_stream.return_value = iter(["streamed ", "content"])

            status, _, _ = self.post_json(
                "/v1/chat/completions",
                {
                    "model": "gemini-3.6-flash",
                    "messages": [{"role": "user", "content": "streaming audit"}],
                    "stream": True,
                },
            )
            self.assertEqual(status, 200)

            files = os.listdir(temp_dir)
            self.assertEqual(len(files), 1)
            with open(os.path.join(temp_dir, files[0]), "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]

            self.assertEqual(len(lines), 1)
            entry = lines[0]
            self.assertEqual(entry["model"], "gemini-3.6-flash")
            self.assertEqual(entry["client_ip"], "127.0.0.1")
            self.assertEqual(entry["request"]["messages"][0]["content"], "streaming audit")
            self.assertEqual(entry["response"]["choices"][0]["message"]["content"], "streamed content")
            self.assertEqual(entry["response"]["choices"][0]["finish_reason"], "stop")

    @mock.patch("gemini_web2api.server.generate", return_value="google content")
    def test_google_generate_content_records_audit_log(self, _generate):
        with tempfile.TemporaryDirectory() as temp_dir:
            CONFIG["audit_log"] = True
            CONFIG["audit_log_dir"] = temp_dir

            status, _, _ = self.post_json(
                "/v1beta/models/gemini-3.6-flash:generateContent",
                {
                    "contents": [{"parts": [{"text": "google prompt"}]}],
                },
            )
            self.assertEqual(status, 200)

            files = os.listdir(temp_dir)
            self.assertEqual(len(files), 1)
            with open(os.path.join(temp_dir, files[0]), "r", encoding="utf-8") as f:
                entry = json.loads(f.readline())

            self.assertEqual(entry["model"], "gemini-3.6-flash")
            self.assertEqual(entry["request"]["contents"][0]["parts"][0]["text"], "google prompt")
            self.assertEqual(entry["response"]["candidates"][0]["content"]["parts"][0]["text"], "google content")

    @mock.patch("gemini_web2api.server.generate", return_value="pacing response")
    def test_server_pacing_rate_limit(self, _generate):
        rate_limiter.reset()
        CONFIG["rate_limit"] = 50.0  # 0.02s
        CONFIG["rate_limit_jitter"] = False
        t0 = time.time()
        self.post_json("/v1/chat/completions", {"model": "gemini-3.6-flash", "messages": [{"role": "user", "content": "1"}]})
        self.post_json("/v1/chat/completions", {"model": "gemini-3.6-flash", "messages": [{"role": "user", "content": "2"}]})
        elapsed = time.time() - t0
        self.assertGreaterEqual(elapsed, 0.015)


class CookieAuthSyncTests(unittest.TestCase):
    def setUp(self):
        self.original_config = dict(CONFIG)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)
        self.temp_dir.cleanup()

    def test_auth_params_not_in_default_config(self):
        self.assertNotIn("gemini_bl", DEFAULT_CONFIG)
        self.assertNotIn("auth_user", DEFAULT_CONFIG)
        self.assertNotIn("xsrf_token", DEFAULT_CONFIG)

    def test_loads_auth_params_directly_from_cookie_file(self):
        cookie_path = os.path.join(self.temp_dir.name, "gemini-auth.json")
        with open(cookie_path, "w", encoding="utf-8") as f:
            json.dump({
                "cookie": "test_cookie",
                "sapisid": "test_sapisid",
                "gemini_bl": "new_bl_from_cookie_2026",
                "auth_user": "2",
                "xsrf_token": "new_xsrf_token_from_cookie",
            }, f)

        CONFIG["cookie_file"] = cookie_path

        url = _get_url()
        self.assertIn("https://gemini.google.com/u/2/_/BardChatUi/data/", url)
        self.assertIn("?bl=new_bl_from_cookie_2026&", url)

        headers = _build_headers()
        self.assertEqual(headers.get("X-Goog-AuthUser"), "2")
        self.assertEqual(headers.get("Referer"), "https://gemini.google.com/u/2/app")
        self.assertEqual(headers.get("Cookie"), "test_cookie")

        payload = _build_payload("hello", 1, 4)
        parsed = parse_qs(payload)
        self.assertEqual(parsed.get("at"), ["new_xsrf_token_from_cookie"])

    def test_falls_back_to_defaults_when_no_cookie_file(self):
        CONFIG["cookie_file"] = None

        url = _get_url()
        self.assertIn("https://gemini.google.com/_/BardChatUi/data/", url)
        self.assertIn(f"?bl={DEFAULT_GEMINI_BL}&", url)

        headers = _build_headers()
        self.assertNotIn("X-Goog-AuthUser", headers)
        self.assertEqual(headers.get("Referer"), "https://gemini.google.com/app")

        payload = _build_payload("hello", 1, 4)
        parsed = parse_qs(payload)
        self.assertNotIn("at", parsed)


class AuditSystemTests(unittest.TestCase):
    def setUp(self):
        self.original_config = dict(CONFIG)
        self.temp_dir = tempfile.TemporaryDirectory()
        CONFIG["audit_log_dir"] = self.temp_dir.name

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)
        self.temp_dir.cleanup()

    def test_default_config_audit_log_disabled(self):
        self.assertIn("audit_log", DEFAULT_CONFIG)
        self.assertIs(DEFAULT_CONFIG["audit_log"], False)

    def test_audit_log_rotation_naming(self):
        dt1 = datetime(2026, 9, 30, 10, 0, 0)
        dt2 = datetime(2026, 10, 1, 10, 0, 0)
        path1 = get_audit_log_path(self.temp_dir.name, dt1)
        path2 = get_audit_log_path(self.temp_dir.name, dt2)

        self.assertEqual(os.path.basename(path1), "audit_2026-09-30.log")
        self.assertEqual(os.path.basename(path2), "audit_2026-10-01.log")

    def test_record_audit_log_when_disabled(self):
        CONFIG["audit_log"] = False
        logged = record_audit_log("127.0.0.1", "gemini-3.8-flash", {"q": "hi"}, {"a": "hello"}, self.temp_dir.name)
        self.assertFalse(logged)
        self.assertEqual(len(os.listdir(self.temp_dir.name)), 0)

    def test_record_audit_log_when_enabled(self):
        CONFIG["audit_log"] = True
        req_obj = {"model": "gemini-3.8-flash", "messages": [{"role": "user", "content": "你好，世界"}]}
        resp_obj = {"choices": [{"message": {"role": "assistant", "content": "你好！有什么我可以帮你的？"}}]}

        logged = record_audit_log("192.168.1.100", "gemini-3.8-flash", req_obj, resp_obj, self.temp_dir.name)
        self.assertTrue(logged)

        files = os.listdir(self.temp_dir.name)
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].startswith("audit_") and files[0].endswith(".log"))

        with open(os.path.join(self.temp_dir.name, files[0]), "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(lines), 1)
        entry = lines[0]
        self.assertEqual(entry["client_ip"], "192.168.1.100")
        self.assertEqual(entry["model"], "gemini-3.8-flash")
        self.assertIn("timestamp", entry)
        self.assertEqual(entry["request"], req_obj)
        self.assertEqual(entry["response"], resp_obj)

    def test_record_audit_log_normalizes_bytes_json(self):
        CONFIG["audit_log"] = True
        req_bytes = json.dumps({"test": "request"}).encode("utf-8")
        resp_bytes = json.dumps({"test": "response"}).encode("utf-8")

        logged = record_audit_log("127.0.0.1", "gemini-3.8-flash", req_bytes, resp_bytes, self.temp_dir.name)
        self.assertTrue(logged)

        files = os.listdir(self.temp_dir.name)
        with open(os.path.join(self.temp_dir.name, files[0]), "r", encoding="utf-8") as f:
            entry = json.loads(f.readline())

        self.assertEqual(entry["request"], {"test": "request"})
        self.assertEqual(entry["response"], {"test": "response"})

    def test_audit_log_thread_safety(self):
        CONFIG["audit_log"] = True
        num_threads = 10
        writes_per_thread = 20

        def worker(tid):
            for i in range(writes_per_thread):
                record_audit_log(
                    f"10.0.0.{tid}",
                    "gemini-3.8-flash",
                    {"thread": tid, "seq": i},
                    {"thread": tid, "reply": i},
                    self.temp_dir.name,
                )

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        files = os.listdir(self.temp_dir.name)
        self.assertEqual(len(files), 1)
        with open(os.path.join(self.temp_dir.name, files[0]), "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(lines), num_threads * writes_per_thread)


class RateLimiterTests(unittest.TestCase):
    def setUp(self):
        self.original_config = dict(CONFIG)
        rate_limiter.reset()

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)
        rate_limiter.reset()

    def test_default_config_rate_limit(self):
        self.assertIn("rate_limit", DEFAULT_CONFIG)
        self.assertIsNone(DEFAULT_CONFIG["rate_limit"])
        self.assertIn("rate_limit_jitter", DEFAULT_CONFIG)
        self.assertIs(DEFAULT_CONFIG["rate_limit_jitter"], False)

    def test_calculate_interval_greater_than_one(self):
        # > 1 represents requests per second
        self.assertAlmostEqual(calculate_interval(2.0, jitter=False), 0.5)
        self.assertAlmostEqual(calculate_interval(5.0, jitter=False), 0.2)
        self.assertAlmostEqual(calculate_interval(10.0, jitter=False), 0.1)

    def test_calculate_interval_fractional_between_zero_and_one(self):
        # 0 < x < 1 represents 1 request every (1/x) seconds
        self.assertAlmostEqual(calculate_interval(0.5, jitter=False), 2.0)
        self.assertAlmostEqual(calculate_interval(0.2, jitter=False), 5.0)
        self.assertAlmostEqual(calculate_interval(0.1, jitter=False), 10.0)

    def test_calculate_interval_disabled(self):
        self.assertEqual(calculate_interval(0, jitter=False), 0.0)
        self.assertEqual(calculate_interval(-1.0, jitter=False), 0.0)
        self.assertEqual(calculate_interval(None, jitter=False), 0.0)

    def test_calculate_interval_with_jitter_range(self):
        # Random deviation within +/- 20% ([0.8, 1.2])
        base_rate = 2.0  # base interval = 0.5s -> [0.4s, 0.6s]
        for _ in range(50):
            interval = calculate_interval(base_rate, jitter=True)
            self.assertGreaterEqual(interval, 0.40 - 1e-6)
            self.assertLessEqual(interval, 0.60 + 1e-6)

        frac_rate = 0.5  # base interval = 2.0s -> [1.6s, 2.4s]
        for _ in range(50):
            interval = calculate_interval(frac_rate, jitter=True)
            self.assertGreaterEqual(interval, 1.60 - 1e-6)
            self.assertLessEqual(interval, 2.40 + 1e-6)

    def test_acquire_disabled_by_default(self):
        CONFIG["rate_limit"] = None
        CONFIG["rate_limit_jitter"] = False
        wait_time = rate_limiter.acquire()
        self.assertEqual(wait_time, 0.0)

    def test_acquire_pacing(self):
        rl = RateLimiter()
        # 50 req/s -> 0.02s interval
        wait1 = rl.acquire(rate=50.0, jitter=False)
        self.assertEqual(wait1, 0.0)

        # Immediate second call should wait ~0.02s
        wait2 = rl.acquire(rate=50.0, jitter=False)
        self.assertGreater(wait2, 0.01)
        self.assertLessEqual(wait2, 0.035)

    def test_rate_limiter_thread_safety(self):
        rl = RateLimiter()
        num_threads = 5
        # 100 req/s -> 0.01s interval
        waits = []
        lock = threading.Lock()

        def worker():
            w = rl.acquire(rate=100.0, jitter=False)
            with lock:
                waits.append(w)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(waits), num_threads)
        self.assertTrue(any(w == 0.0 for w in waits))
        self.assertTrue(any(w > 0.0 for w in waits))


class ModelResolutionTests(unittest.TestCase):
    def test_gemini_4_argon_in_modular_models(self):
        from gemini_web2api.models import MODELS, resolve_model

        self.assertIn("gemini-4-argon", MODELS)
        self.assertEqual(MODELS["gemini-4-argon"]["mode"], 1)
        self.assertEqual(MODELS["gemini-4-argon"]["think"], 4)

        self.assertIn("gemini-4-argon-high", MODELS)
        self.assertEqual(MODELS["gemini-4-argon-high"]["mode"], 2)
        self.assertEqual(MODELS["gemini-4-argon-high"]["think"], 0)

        name, mode_id, think_mode, err, extra = resolve_model("gemini-4-argon")
        self.assertEqual((name, mode_id, think_mode, err, extra), ("gemini-4-argon", 1, 4, None, None))

        name_h, mode_h, think_h, err_h, extra_h = resolve_model("gemini-4-argon-high")
        self.assertEqual((name_h, mode_h, think_h, err_h, extra_h), ("gemini-4-argon-high", 2, 0, None, None))

class MainEntrypointTests(unittest.TestCase):
    def test_import_main_entrypoint(self):
        from gemini_web2api.main import main
        from gemini_web2api import main as main_module

        self.assertTrue(callable(main))
        self.assertIs(main, main_module.main)

    def test_main_cli_version(self):
        import subprocess
        import sys

        cmd = [sys.executable, "gemini_web2api/main.py", "--version"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("gemini-web2api", res.stdout + res.stderr)

    def test_module_main_cli_version(self):
        import subprocess
        import sys

        cmd = [sys.executable, "-m", "gemini_web2api.main", "--version"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("gemini-web2api", res.stdout + res.stderr)

    def test_find_config_locates_package_config(self):
        from gemini_web2api.config import find_config
        cfg = find_config()
        self.assertIsNotNone(cfg)
        self.assertTrue(os.path.exists(cfg))
        self.assertTrue(cfg.endswith("config.json"))

    def test_resolve_cookie_file(self):
        from gemini_web2api.config import resolve_cookie_file
        resolved = resolve_cookie_file("gemini-auth.json")
        self.assertIsNotNone(resolved)
        self.assertTrue(os.path.isabs(resolved) or os.path.exists(resolved))


if __name__ == "__main__":
    unittest.main()
