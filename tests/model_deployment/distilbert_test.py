"""Capture FP32 HTTP reference, then test deployment accuracy and concurrent latency."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from tests.llm_e2e.metrics import percentile

TEXTS = [
    ['The quick brown fox jumps over the lazy dog.'],
    ['Hello world.', 'A GPU processes many arithmetic operations in parallel.'],
    ['A short sentence.', 'This is a longer document. ' * 35],
    ['Testing model deployment, padding, and tokenization.'] * 8,
    ['Numbers: 0, -1, 42. Unicode: café, naïve.'],
    ['attention ' * 126],
    ['attention ' * 510],
    ['A detailed log of inference events. ' * 25] * 4,
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8001')
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference', help='FP32 references.json')
    parser.add_argument('--requests', type=int, default=32)
    parser.add_argument('--atol', type=float)
    parser.add_argument('--rtol', type=float)
    args = parser.parse_args()
    import numpy as np
    import requests
    if args.requests < 8 or (args.atol is not None and args.atol < 0) or (args.rtol is not None and args.rtol < 0):
        parser.error('requests >= 8 and non-negative tolerances required')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    info_response = requests.get(args.url + '/health', timeout=30)
    info_response.raise_for_status()
    info = info_response.json()
    data_hash = hashlib.sha256(json.dumps(TEXTS).encode()).hexdigest()
    reference = json.loads(Path(args.reference).read_text()) if args.reference else None
    if reference is None and info['mode'] != 'fp32':
        raise ValueError('capture reference with FP32 first')
    if reference and (reference['info']['model_hash'] != info['model_hash'] or reference['data_hash'] != data_hash or reference['info']['mode'] != 'fp32'):
        raise ValueError('reference model/data/precision mismatch')
    configured = args.atol is not None and args.rtol is not None and np.isfinite([args.atol, args.rtol]).all()
    reference_rows, comparisons = (reference['rows'] if reference else []), []
    status, failures = 'PASS', []
    try:
        for i, texts in enumerate(TEXTS):
            response = requests.post(args.url + '/encode', json={'texts': texts}, timeout=90)
            response.raise_for_status()
            row = response.json()
            arr = np.asarray(row['hidden_states'])
            assert row['model_hash'] == info['model_hash'] and row['mode'] == info['mode']
            assert np.isfinite(arr).all() and arr.shape == (len(texts), len(row['input_ids'][0]), 768)
            if not reference:
                reference_rows.append(row)
            if reference:
                golden = reference['rows'][i]
                assert row['input_ids'] == golden['input_ids'] and row['attention_mask'] == golden['attention_mask']
                mask = np.asarray(row['attention_mask'], dtype=bool)
                actual, expected = arr[mask], np.asarray(golden['hidden_states'])[mask]
                delta = actual - expected
                close = bool(np.allclose(actual, expected, atol=args.atol, rtol=args.rtol)) if configured else None
                comparisons.append({'case': i, 'max_abs': float(np.max(np.abs(delta))),
                                    'mean_abs': float(np.mean(np.abs(delta))),
                                    'relative_l2': float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-12)),
                                    'cosine': float(np.sum(actual * expected) / max(np.linalg.norm(actual) * np.linalg.norm(expected), 1e-12)),
                                    'within_tolerance': close})
                if close is False:
                    failures.append(f'accuracy case {i}')
        if not reference:
            (output / 'references.json').write_text(json.dumps({'info': info, 'data_hash': data_hash, 'rows': reference_rows}))
        for bad in ({'texts': []}, {'texts': ['attention ' * 512]}, {'texts': ['ok'] * 9}):
            response = requests.post(args.url + '/encode', json=bad, timeout=90)
            if response.status_code != 400:
                failures.append('negative case rejected incorrectly: ' + str(response.status_code))
        recovery = requests.post(args.url + '/encode', json={'texts': ['Recovery request.']}, timeout=90)
        recovery.raise_for_status()
        groups = {}
        raw_file = output / 'requests.jsonl'
        for concurrency in (1, 2, 4, 8):
            def one(index):
                began = time.perf_counter()
                try:
                    case = index % len(TEXTS)
                    response = requests.post(args.url + '/encode', json={'texts': TEXTS[case]}, timeout=90)
                    response.raise_for_status()
                    value = response.json()
                    arr = np.asarray(value['hidden_states'])
                    golden = reference_rows[case]
                    ok = (value['model_hash'] == info['model_hash'] and value['mode'] == info['mode']
                          and value['input_ids'] == golden['input_ids']
                          and value['attention_mask'] == golden['attention_mask']
                          and arr.shape == np.asarray(golden['hidden_states']).shape and np.isfinite(arr).all()
                          and np.allclose(arr, golden['hidden_states'], atol=args.atol or 0, rtol=args.rtol or 0))
                    return {'index': index, 'case': case, 'concurrency': concurrency, 'ok': bool(ok),
                            'request_id': value['request_id'], 'e2e_ms': (time.perf_counter() - began) * 1000,
                            'forward_and_copy_ms': value['forward_and_copy_ms'],
                            'torch_peak_allocated_bytes': value['torch_peak_allocated_bytes']}
                except Exception as exc:
                    return {'index': index, 'concurrency': concurrency, 'ok': False, 'error': str(exc),
                            'e2e_ms': (time.perf_counter() - began) * 1000}
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                rows = list(pool.map(one, range(args.requests)))
            elapsed = time.perf_counter() - start
            good = [r for r in rows if r['ok']]
            ids = [r['request_id'] for r in good]
            if len(good) != len(rows) or len(set(ids)) != len(ids):
                failures.append(f'concurrency {concurrency}: failed or duplicate response')
            groups[str(concurrency)] = {'requests': len(rows), 'success_rate': len(good) / len(rows),
                                        'requests_per_s': len(good) / elapsed,
                                        'e2e_p50_ms': percentile([r['e2e_ms'] for r in good], .5),
                                        'e2e_p95_ms': percentile([r['e2e_ms'] for r in good], .95)}
            with raw_file.open('a') as handle:
                for row in rows:
                    handle.write(json.dumps(row) + '\n')
            print('concurrency', concurrency, groups[str(concurrency)], flush=True)
        status = 'BLOCKED' if failures else ('REFERENCE_CAPTURED' if not reference else ('CORRECTNESS_PASS' if configured else 'NEEDS_TOLERANCE'))
    except Exception as exc:
        status = 'BLOCKED'
        failures.append(f'{type(exc).__name__}: {exc}')
        groups = locals().get('groups', {})
    summary = {'status': status, 'scope': 'correctness gate + performance collection; no performance SLA gate',
               'info': info, 'data_hash': data_hash, 'accuracy': comparisons, 'performance': groups,
               'atol': args.atol, 'rtol': args.rtol, 'failures': failures}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))
    return 0 if status in ('CORRECTNESS_PASS', 'REFERENCE_CAPTURED') else 2


if __name__ == '__main__':
    raise SystemExit(main())
