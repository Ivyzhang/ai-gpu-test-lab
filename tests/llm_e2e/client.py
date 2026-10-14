"""Small OpenAI-compatible HTTP client with client-observed stream timing."""
from contextlib import contextmanager
import http.client
import ipaddress
import json
import os
import socket
import threading
import time
from urllib.parse import urlsplit
import uuid


class StreamAccumulator:
    def __init__(self, started):
        self.started = started
        self.first = self.last = self.completed = None
        self.parts = []
        self.gaps_ms = []
        self.finish = None
        self.usage = {}
        self.response_id = None

    def feed(self, data, at):
        if data == '[DONE]':
            self.completed = at
            return
        value = json.loads(data)
        if value.get('error'):
            raise ValueError('server stream error: ' + str(value['error']))
        rid = value.get('id')
        if rid:
            if self.response_id and self.response_id != rid:
                raise ValueError('response ID changed within one stream')
            self.response_id = rid
        if value.get('usage'):
            self.usage = value['usage']
        choices = value.get('choices', [])
        if len(choices) > 1:
            raise ValueError('expected one completion choice')
        for choice in choices:
            if choice.get('index', 0) != 0:
                raise ValueError('unexpected choice index')
            if choice.get('finish_reason') is not None:
                self.finish = choice['finish_reason']
            content = choice.get('delta', {}).get('content')
            if content:
                if not isinstance(content, str):
                    raise ValueError('delta.content must be text')
                if self.first is None:
                    self.first = at
                if self.last is not None:
                    self.gaps_ms.append((at - self.last) * 1000)
                self.last = at
                self.parts.append(content)

    def result(self):
        tokens = self.usage.get('completion_tokens')
        prompt = self.usage.get('prompt_tokens')
        if type(tokens) is not int or tokens < 1:
            tokens = None
        if type(prompt) is not int or prompt < 1:
            prompt = None
        ok = self.first is not None and self.completed is not None and self.finish in ('stop', 'length')
        return {
            'ok': ok, 'error': None if ok else 'incomplete/empty stream or unsupported finish_reason',
            'text': ''.join(self.parts), 'response_id': self.response_id,
            'finish_reason': self.finish, 'content_chunks': len(self.parts),
            'chunk_gaps_ms': self.gaps_ms,
            'prompt_tokens': prompt, 'completion_tokens': tokens,
            'ttft_ms': None if self.first is None else (self.first - self.started) * 1000,
            'e2e_ms': None if self.completed is None else (self.completed - self.started) * 1000,
            'last_content_ms': None if self.last is None else (self.last - self.started) * 1000,
            'tpot_estimate_ms': ((self.last - self.first) * 1000 / (tokens - 1)
                                 if tokens and tokens > 1 and len(self.parts) > 1 else None),
        }


def _shutdown(sock):
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


class Client:
    def __init__(self, base_url, timeout_s=60):
        parsed = urlsplit(base_url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.query or parsed.fragment:
            raise ValueError('base_url must be an http(s) URL without credentials/query/fragment')
        self.connect_host = '127.0.0.1' if parsed.hostname == 'localhost' else parsed.hostname
        try:
            ipaddress.ip_address(self.connect_host)
        except ValueError as exc:
            raise ValueError('use localhost or a numeric IP; DNS hostnames have no bounded resolution deadline') from exc
        self.base_url = base_url.rstrip('/')
        self.parsed = parsed
        self.timeout_s = timeout_s

    @contextmanager
    def connection(self, method, path, payload=None):
        kind = http.client.HTTPSConnection if self.parsed.scheme == 'https' else http.client.HTTPConnection
        conn = kind(self.connect_host, self.parsed.port, timeout=self.timeout_s)
        started = time.perf_counter()
        timer = None
        try:
            conn.connect()
            remaining = self.timeout_s - (time.perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError('connect exceeded request deadline')
            timer = threading.Timer(remaining, _shutdown, args=(conn.sock,))
            timer.daemon = True
            timer.start()
            headers = {'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream'}
            key = os.environ.get('LLM_API_KEY')
            if key:
                headers['Authorization'] = 'Bearer ' + key
            body = None if payload is None else json.dumps(payload).encode('utf-8')
            conn.request(method, self.parsed.path.rstrip('/') + path, body=body, headers=headers)
            response = conn.getresponse()
            yield response
        finally:
            if timer:
                timer.cancel()
            conn.close()

    def json_request(self, method, path, payload=None):
        with self.connection(method, path, payload) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError('JSON response too large')
            return response.status, json.loads(raw.decode('utf-8'))

    def generate(self, model, prompt, max_tokens=64, stream=True):
        request_id = str(uuid.uuid4())
        started_unix_s = time.time()
        started = time.perf_counter()
        state = StreamAccumulator(started)
        payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                   'max_tokens': max_tokens, 'temperature': 0, 'stream': stream, 'n': 1}
        if stream:
            payload['stream_options'] = {'include_usage': True}
        result = {'ok': False}
        status = None
        try:
            with self.connection('POST', '/v1/chat/completions', payload) as response:
                status = response.status
                if status != 200:
                    raise ValueError(f'HTTP {status}: ' + response.read(2048).decode('utf-8', errors='replace'))
                if not stream:
                    value = json.loads(response.read(4 * 1024 * 1024).decode('utf-8'))
                    choice = value['choices'][0]
                    text = choice['message'].get('content', '')
                    usage = value.get('usage') or {}
                    result = {'ok': isinstance(text, str) and bool(text.strip()) and choice.get('finish_reason') in ('stop', 'length'),
                              'text': text, 'response_id': value.get('id'),
                              'finish_reason': choice.get('finish_reason'),
                              'prompt_tokens': usage.get('prompt_tokens'), 'completion_tokens': usage.get('completion_tokens'),
                              'ttft_ms': None, 'tpot_estimate_ms': None,
                              'e2e_ms': (time.perf_counter() - started) * 1000}
                else:
                    if 'text/event-stream' not in response.getheader('Content-Type', ''):
                        raise ValueError('streaming request did not return text/event-stream')
                    data_lines = []
                    received = 0
                    while state.completed is None:
                        raw = response.readline(1024 * 1024 + 1)
                        if not raw:
                            break
                        received += len(raw)
                        if len(raw) > 1024 * 1024 or received > 16 * 1024 * 1024:
                            raise ValueError('SSE size limit exceeded')
                        line = raw.decode('utf-8').rstrip('\r\n')
                        if not line:
                            if data_lines:
                                state.feed('\n'.join(data_lines), time.perf_counter())
                                data_lines = []
                        elif line.startswith('data:'):
                            data_lines.append(line[5:].lstrip(' '))
                    result = state.result()
            tokens = result.get('completion_tokens')
            if tokens is not None and (type(tokens) is not int or tokens < 1 or tokens > max_tokens):
                result.update(ok=False, error='completion_tokens violates max_tokens contract')
        except Exception as exc:
            result = state.result() if stream else result
            result.update(ok=False, error=f'{type(exc).__name__}: {exc}')
        result.update(request_id=request_id, http_status=status, stream=stream,
                      started_unix_s=started_unix_s,
                      wall_ms=(time.perf_counter() - started) * 1000, requested_max_tokens=max_tokens)
        return result
