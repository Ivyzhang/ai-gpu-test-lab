"""Real loopback HTTP/SSE integration; the server is a protocol fixture, NOT a GPU model."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import threading
import time
import unittest
import uuid

from tests.llm_e2e.client import Client
from tests.llm_e2e.run import execute


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.send_json(200, {'data': [{'id': 'protocol-fixture'}]})

    def do_POST(self):
        value = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        model = value['model']
        if model == '__missing_model__':
            return self.send_json(404, {'error': {'message': 'unknown model'}})
        if value['max_tokens'] < 1:
            return self.send_json(400, {'error': {'message': 'max_tokens must be positive'}})
        if model == 'slow':
            time.sleep(.4)
            return
        prompt = value['messages'][0]['content']
        match = re.search(r'What is (\d+) \+ 1', prompt)
        pieces = [str(int(match[1]) + 1)] if match else ['Hello', ' world']
        pieces = pieces[:value['max_tokens']]
        rid = 'chatcmpl-' + str(uuid.uuid4())
        usage = {'prompt_tokens': 12, 'completion_tokens': len(pieces), 'total_tokens': 12 + len(pieces)}
        if not value.get('stream'):
            return self.send_json(200, {'id': rid, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': ''.join(pieces)},
                                                              'finish_reason': 'stop'}], 'usage': usage})
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        def emit(obj):
            data = obj if isinstance(obj, str) else json.dumps(obj)
            self.wfile.write(('data: ' + data + '\n\n').encode())
            self.wfile.flush()
        try:
            self.wfile.write(b': heartbeat\n\n')
            emit({'id': rid, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})
            for part in pieces:
                time.sleep(.01)
                emit({'id': rid, 'choices': [{'index': 0, 'delta': {'content': part}, 'finish_reason': None}]})
            emit({'id': rid, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})
            if model != 'no-usage':
                emit({'id': rid, 'choices': [], 'usage': usage})
            if model != 'truncated':
                emit('[DONE]')
        except (BrokenPipeError, ConnectionResetError):
            pass


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f'http://127.0.0.1:{cls.server.server_address[1]}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_sse_roundtrip_reconstructs_text_and_usage(self):
        row = Client(self.base_url, 2).generate('protocol-fixture', 'hello')
        self.assertTrue(row['ok'], row)
        self.assertEqual(row['text'], 'Hello world')
        self.assertEqual(row['completion_tokens'], 2)
        self.assertGreater(row['e2e_ms'], row['ttft_ms'])

    def test_non_stream_roundtrip_has_no_fake_ttft(self):
        row = Client(self.base_url, 2).generate('protocol-fixture', 'hello', stream=False)
        self.assertTrue(row['ok'], row)
        self.assertIsNone(row['ttft_ms'])

    def test_http_rejection_is_preserved(self):
        row = Client(self.base_url, 2).generate('__missing_model__', 'hello')
        self.assertFalse(row['ok'])
        self.assertEqual(row['http_status'], 404)
        self.assertIn('unknown model', row['error'])

    def test_truncated_http_stream_is_failure(self):
        row = Client(self.base_url, 2).generate('truncated', 'hello')
        self.assertFalse(row['ok'])

    def test_missing_usage_remains_missing(self):
        row = Client(self.base_url, 2).generate('no-usage', 'hello')
        self.assertTrue(row['ok'])
        self.assertIsNone(row['completion_tokens'])

    def test_deadline_aborts_slow_response(self):
        started = time.perf_counter()
        row = Client(self.base_url, .05).generate('slow', 'hello')
        self.assertFalse(row['ok'])
        self.assertLess(time.perf_counter() - started, .3)

    def test_suite_writes_raw_evidence_and_passes_configured_gate(self):
        config = {'base_url': self.base_url, 'served_model_id': None,
                  'request_timeout_s': 2, 'readiness_timeout_s': 2,
                  'load': {'scenarios': ['short', 'mixed'], 'concurrency': [1, 2],
                           'requests_per_case': 4, 'warmup_per_case': 1, 'max_tokens': 8},
                  'gate': {'min_samples': 4, 'min_success_rate': 1.0, 'ttft_p95_ms': 500,
                           'e2e_p95_ms': 1000, 'tpot_estimate_p95_ms': 500, 'max_regression_fraction': .1}}
        deployment = {'allowed_model_ids': ['protocol-fixture'], 'state': 'ready',
                      'server_host': '127.0.0.1',
                      'server_pid': os.getpid(), 'launch_hostname': platform.node(), 'server_port': self.server.server_address[1],
                      'controlled': {'model_hash': 'TEST-ONLY-NO-MODEL', 'gpu': 'TEST-ONLY-NO-GPU',
                                     'server_command': ['protocol-fixture'], 'cache_policy': 'none'}}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'test-run'
            report = execute(config, deployment, out)
            self.assertEqual(report['gate']['status'], 'PASS', report)
            rows = [json.loads(line) for line in (out / 'requests.jsonl').read_text().splitlines()]
            measured = [r for r in rows if r['phase'] == 'measured']
            self.assertEqual(len(measured), 16)
            self.assertEqual(len({r['request_id'] for r in rows}), len(rows))
            self.assertTrue((out / 'report.md').exists())
            self.assertTrue((out / 'summary.json').exists())

    def test_wrong_model_identity_blocks_before_benchmark(self):
        config = {'base_url': self.base_url, 'request_timeout_s': 2, 'readiness_timeout_s': 2,
                  'load': {'scenarios': ['short'], 'concurrency': [1], 'requests_per_case': 2,
                           'warmup_per_case': 1, 'max_tokens': 8},
                  'gate': {'min_samples': 2, 'min_success_rate': 1, 'ttft_p95_ms': 500,
                           'e2e_p95_ms': 1000, 'tpot_estimate_p95_ms': 500}}
        with tempfile.TemporaryDirectory() as temp:
            r = execute(config, {'allowed_model_ids': ['different-model'],
                                'server_host': '127.0.0.1',
                                'state': 'ready', 'server_pid': os.getpid(), 'launch_hostname': platform.node(),
                                'server_port': self.server.server_address[1],
                                'controlled': {'model_hash': 'fixture', 'gpu': 'fixture', 'server_command': ['fixture'], 'cache_policy': 'none'}},
                        Path(temp) / 'run')
            self.assertEqual(r['gate']['status'], 'BLOCKED')
            self.assertEqual(r['cases'], {})

    def test_client_rejects_dns_hostname_before_request(self):
        with self.assertRaises(ValueError):
            Client('http://unresolvable.example:8000', .1)


if __name__ == '__main__':
    unittest.main()
