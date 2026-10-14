"""Run functional checks and closed-loop concurrency tests against a real server."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

from .client import Client
from .deployment import verify_live_deployment
from .metrics import evaluate, summarize


def parallel_requests(call, count, concurrency):
    if type(count) is not int or type(concurrency) is not int or count < 1 or concurrency < 1:
        raise ValueError('count and concurrency must be positive integers')
    def safe(index):
        try:
            return call(index)
        except Exception as exc:
            return {'ok': False, 'index': index, 'error': f'{type(exc).__name__}: {exc}'}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(safe, i) for i in range(count)]
        rows = [future.result() for future in as_completed(futures)]
    return rows, time.perf_counter() - started


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def append_rows(path, rows):
    with path.open('a', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def prompt_for(kind, index):
    short = f'Request {index}. Explain in English how to prepare a cup of tea, with numbered steps.'
    long = '\n'.join(f'Record {n}: server {n % 7} handled a request successfully in {10 + n % 11} milliseconds.' for n in range(160))
    long = f'Request {index}. Read these logs and summarize their common pattern:\n{long}'
    if kind == 'short':
        return short
    if kind == 'long':
        return long
    if kind == 'mixed':
        return short if index % 2 == 0 else long
    raise ValueError('unsupported scenario: ' + kind)


def wait_ready(client, configured_model, seconds, process_alive=None):
    deadline = time.perf_counter() + seconds
    error = 'no attempt'
    probe = Client(client.base_url, timeout_s=min(5, seconds))
    while time.perf_counter() < deadline:
        if process_alive is not None and not process_alive():
            raise RuntimeError('server process exited before readiness')
        try:
            status, value = probe.json_request('GET', '/v1/models')
            ids = [row['id'] for row in value.get('data', [])]
            if status == 200:
                if configured_model and configured_model in ids:
                    return configured_model
                if not configured_model and len(ids) == 1:
                    return ids[0]
            error = f'status={status}, model IDs={ids}; set served_model_id if multiple models'
        except Exception as exc:
            error = str(exc)
        time.sleep(min(1, max(0, deadline - time.perf_counter())))
    raise RuntimeError('model readiness timed out: ' + error)


def functional_checks(client, model, raw_path):
    checks = []
    def check(name, passed, detail):
        checks.append({'name': name, 'ok': bool(passed), 'detail': detail})

    for stream in (False, True):
        row = client.generate(model, 'Reply with a short greeting in English.', 32, stream=stream)
        append_rows(raw_path, [dict(row, phase='functional')])
        check('stream' if stream else 'non_stream', row['ok'], row.get('error'))

    row = client.generate(model, '你好，请用中文简单介绍你自己。', 32)
    append_rows(raw_path, [dict(row, phase='unicode')])
    check('unicode_transport', row['ok'] and bool(row.get('text')), row.get('error'))

    row = client.generate(model, 'Count upwards from 1 and keep going.', 1)
    append_rows(raw_path, [dict(row, phase='max_tokens')])
    check('max_tokens_one', row['ok'] and row.get('completion_tokens') == 1, row.get('error'))

    for name, payload, expected in [
        ('unknown_model', {'model': '__missing_model__', 'messages': [{'role': 'user', 'content': 'hello'}], 'max_tokens': 2}, {400, 404, 422}),
        ('negative_max_tokens', {'model': model, 'messages': [{'role': 'user', 'content': 'hello'}], 'max_tokens': -1}, {400, 422}),
    ]:
        try:
            status, value = client.json_request('POST', '/v1/chat/completions', payload)
            check(name, status in expected, {'status': status, 'body': value})
        except Exception as exc:
            check(name, False, str(exc))
    row = client.generate(model, 'Say hello.', 16)
    append_rows(raw_path, [dict(row, phase='recovery')])
    check('valid_after_invalid', row['ok'], row.get('error'))

    def arithmetic(i):
        target = i + 2
        row = client.generate(model, f'What is {i + 1} + 1? Reply with only the integer, no explanation.', 16)
        row.update(phase='parallel_arithmetic', target=target, index=i)
        row['task_ok'] = bool(row['ok'] and re.fullmatch(r'\s*' + str(target) + r'\s*', row.get('text', '')))
        return row
    rows, _ = parallel_requests(arithmetic, count=8, concurrency=4)
    append_rows(raw_path, rows)
    check('parallel_arithmetic', all(r.get('task_ok') for r in rows),
          '8 distinct tasks, concurrency 4; failure may be model instruction-following or response isolation: inspect raw responses')
    response_ids = [r.get('response_id') for r in rows]
    check('parallel_response_ids', all(response_ids) and len(set(response_ids)) == len(rows), 'completion IDs must be unique')
    return checks


def validate_config(config):
    load = config['load']
    for key in ('requests_per_case', 'warmup_per_case', 'max_tokens'):
        if type(load.get(key)) is not int or load[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    levels = load.get('concurrency', [])
    if not levels or len(set(levels)) != len(levels) or any(type(x) is not int or x < 1 or x > load['requests_per_case'] for x in levels):
        raise ValueError('concurrency must be unique positive integers <= requests_per_case')
    scenarios = load.get('scenarios', [])
    if not scenarios or len(set(scenarios)) != len(scenarios) or not set(scenarios) <= {'short', 'long', 'mixed'}:
        raise ValueError('scenarios must be unique values from short, long, mixed')
    for key in ('request_timeout_s', 'readiness_timeout_s'):
        value = config.get(key)
        if type(value) not in (float, int) or not 0 < value < 86400:
            raise ValueError(key + ' must be positive and finite, below 86400')
    policy = config['gate']
    if type(policy.get('min_samples')) is not int or not 1 <= policy['min_samples'] <= load['requests_per_case']:
        raise ValueError('min_samples must be between 1 and requests_per_case')
    if type(policy.get('min_success_rate')) not in (float, int) or not 0 < policy['min_success_rate'] <= 1:
        raise ValueError('min_success_rate must be in (0, 1]')


def execute(config, deployment, output, baseline=None, require_baseline=False):
    validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    load = config['load']
    report = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'valid': False, 'functional': [], 'cases': {},
              'expected_cases': [f'{s}-c{c}' for s in load['scenarios'] for c in load['concurrency']],
              'controlled': {}, 'config': config, 'deployment': deployment,
              'scope': 'real HTTP service; closed-loop concurrency; no engine-side ITL or open-loop SLA claim'}
    save(output / 'config.json', config)
    save(output / 'deployment.json', deployment)
    raw_path = output / 'requests.jsonl'
    try:
        controls = deployment['controlled']
        if not all(controls.get(k) for k in ('model_hash', 'gpu', 'server_command', 'cache_policy')):
            raise ValueError('deployment manifest lacks required controlled fields')
        report['attribution_before'] = verify_live_deployment(deployment, config['base_url'])
        report['controlled'] = dict(controls, load_hash=digest(load),
                                    client_protocol_version=1, base_url=config['base_url'],
                                    prompt_dataset_hash=digest([prompt_for(s, i) for s in load['scenarios'] for i in range(load['requests_per_case'])]))
        client = Client(config['base_url'], config['request_timeout_s'])
        started = time.perf_counter()
        model = wait_ready(client, config.get('served_model_id'), config['readiness_timeout_s'])
        report['readiness_wait_ms'] = (time.perf_counter() - started) * 1000
        report['served_model_id'] = model
        allowed = deployment.get('allowed_model_ids', [])
        if model not in allowed:
            raise ValueError(f'served model {model!r} not in deployment.allowed_model_ids; verify model identity')
        report['functional'] = functional_checks(client, model, raw_path)
        for scenario in load['scenarios']:
            for concurrency in load['concurrency']:
                name = f'{scenario}-c{concurrency}'
                for i in range(load['warmup_per_case']):
                    warm = client.generate(model, prompt_for(scenario, -i - 1), load['max_tokens'])
                    append_rows(raw_path, [dict(warm, phase='warmup', case=name)])
                    if not warm['ok']:
                        raise RuntimeError(f'{name} warmup failed: {warm.get("error")}')
                def call(i):
                    row = client.generate(model, prompt_for(scenario, i), load['max_tokens'])
                    return dict(row, phase='measured', case=name, index=i,
                                prompt_class=('short' if i % 2 == 0 else 'long') if scenario == 'mixed' else scenario)
                rows, elapsed = parallel_requests(call, load['requests_per_case'], concurrency)
                append_rows(raw_path, rows)
                stats = summarize(rows, elapsed)
                if scenario == 'mixed':
                    stats['by_prompt_class'] = {kind: summarize([r for r in rows if r.get('prompt_class') == kind], elapsed) for kind in ('short', 'long')}
                report['cases'][name] = stats
                print(f'{name}: {stats["successes"]}/{stats["requests"]}, TTFT p95={stats["ttft_p95_ms"]}', flush=True)
                save(output / 'partial-summary.json', report)
        report['attribution_after'] = verify_live_deployment(deployment, config['base_url'])
        report['valid'] = True
    except Exception as exc:
        report['infrastructure_error'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['gate'] = evaluate(report, config['gate'], baseline, require_baseline)
        save(output / 'summary.json', report)
        (output / 'report.md').write_text(markdown_report(report), encoding='utf-8')
    return report


def markdown_report(report):
    lines = ['# Qwen TensorRT-LLM 服务测试报告', '', '**状态：' + report['gate']['status'] + '**', '',
             '客户端观测；TPOT 为估计值；p99 是描述值，样本不足时不能作为尾延迟验收证据。', '',
             '| 场景 | 成功/总数 | TTFT p95 ms | E2E p95 ms | TPOT估计 p95 ms | output tokens/s |',
             '|---|---:|---:|---:|---:|---:|']
    def display(x):
        return 'N/A' if x is None else f'{x:.2f}'
    for name, s in report['cases'].items():
        lines.append(f'| {name} | {s["successes"]}/{s["requests"]} | ' + ' | '.join(display(s[k]) for k in ('ttft_p95_ms', 'e2e_p95_ms', 'tpot_estimate_p95_ms', 'output_tokens_per_s')) + ' |')
    lines.extend(['', '## 判定原因', ''] + ['- ' + reason for reason in report['gate']['reasons']])
    if report.get('infrastructure_error'):
        lines.extend(['', '基础设施/预检：' + report['infrastructure_error']])
    lines.extend(['', '## 功能用例', ''] + [f'- {x["name"]}: {"PASS" if x["ok"] else "FAIL"}' for x in report['functional']])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.example.json')))
    parser.add_argument('--deployment', required=True, help='deployment.json created by tests.llm_e2e.serve')
    parser.add_argument('--output', required=True, help='new report directory; never overwrite earlier evidence')
    parser.add_argument('--baseline', help='reviewed summary.json from a comparable earlier run')
    parser.add_argument('--require-baseline', action='store_true')
    args = parser.parse_args()
    try:
        config = json.loads(Path(args.config).read_text())
        deployment = json.loads(Path(args.deployment).read_text())
        baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
        report = execute(config, deployment, Path(args.output), baseline, args.require_baseline)
        print(json.dumps(report['gate'], ensure_ascii=False, indent=2))
        return 0 if report['gate']['status'] == 'PASS' else 2
    except Exception as exc:
        print(f'Configuration/report setup failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
