"""Small fixed task smoke set: compare quantized Qwen with the same model's FP16 run."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from tests.llm_e2e.client import Client
from tests.llm_e2e.deployment import verify_live_deployment
from tests.llm_e2e.run import wait_ready


def cases():
    result = []
    for i in range(24):
        a, b = i + 3, i % 7 + 2
        result.append({'id': f'add-{i}', 'prompt': f'Calculate {a} + {b}. Reply with only the integer answer.', 'expected': str(a + b)})
        result.append({'id': f'extract-{i}', 'prompt': f'The order ID is {7000 + i}. The quantity is 3. Reply with only the order ID.', 'expected': str(7000 + i)})
    return result


def correct(text, expected):
    return isinstance(text, str) and re.fullmatch(r'\s*' + re.escape(expected) + r'\s*', text) is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference', help='FP16 quality.json captured by this tool')
    parser.add_argument('--max-accuracy-drop-pp', type=float, help='approved percentage-point budget, e.g. 1 means one percentage point')
    args = parser.parse_args()
    if args.max_accuracy_drop_pp is not None and not 0 <= args.max_accuracy_drop_pp <= 100:
        parser.error('accuracy drop budget must be between 0 and 100')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    deployment = json.loads(Path(args.deployment).read_text())
    data = cases()
    data_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    reference = json.loads(Path(args.reference).read_text()) if args.reference else None
    rows, failures, status = [], [], 'BLOCKED'
    try:
        control = deployment['controlled']
        if control.get('runtime') != 'vllm' or control.get('dtype') != 'half':
            raise ValueError('use the vLLM launcher with --dtype half for both reference and candidate')
        if not reference and control.get('quantization') != 'none':
            raise ValueError('capture an unquantized FP16 quality reference first')
        if reference:
            ref_control = reference['deployment']['controlled']
            if (reference.get('data_hash') != data_hash or reference.get('transport_success_rate') != 1
                    or reference.get('status') != 'REFERENCE_CAPTURED' or ref_control.get('quantization') != 'none'
                    or ref_control.get('model_hash') != control['model_hash'] or ref_control.get('dtype') != 'half'
                    or len(reference['rows']) != len(data)):
                raise ValueError('reference identity, dataset, dtype, or completeness mismatch')
        verify_live_deployment(deployment, args.url)
        client = Client(args.url, 90)
        model = wait_ready(client, None, 30)
        if model not in deployment['allowed_model_ids']:
            raise ValueError('wrong served model')
        for case in data:
            row = client.generate(model, case['prompt'], max_tokens=32, stream=False)
            row.update(case_id=case['id'], expected=case['expected'],
                       task_correct=bool(row['ok'] and correct(row.get('text'), case['expected'])))
            rows.append(row)
            with (output / 'responses.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
        verify_live_deployment(deployment, args.url)
        if not all(row['ok'] for row in rows):
            failures.append('transport or generation failures; inspect responses.jsonl')
        if failures:
            status = 'BLOCKED'
        elif not reference:
            status = 'REFERENCE_CAPTURED'
        elif args.max_accuracy_drop_pp is None:
            status = 'NEEDS_QUALITY_BUDGET'
        else:
            accuracy = sum(r['task_correct'] for r in rows) / len(data)
            drop = (reference['accuracy'] - accuracy) * 100
            status = 'QUALITY_SMOKE_PASS' if drop <= args.max_accuracy_drop_pp else 'BLOCKED'
            if status == 'BLOCKED':
                failures.append(f'accuracy drop {drop:.3f} pp exceeds budget {args.max_accuracy_drop_pp}')
    except Exception as exc:
        failures.append(f'{type(exc).__name__}: {exc}')
    accuracy = sum(r['task_correct'] for r in rows) / len(data)
    report = {'status': status, 'scope': '48 synthetic arithmetic/extraction tasks only; NOT comprehensive LLM evaluation',
              'deployment': deployment, 'data_hash': data_hash, 'rows': rows,
              'transport_success_rate': sum(r['ok'] for r in rows) / len(data), 'accuracy': accuracy,
              'baseline_accuracy': reference.get('accuracy') if reference else None,
              'accuracy_drop_pp': (reference['accuracy'] - accuracy) * 100 if reference else None,
              'max_accuracy_drop_pp': args.max_accuracy_drop_pp, 'failures': failures}
    (output / 'quality.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'deployment')}, indent=2))
    return 0 if status in ('REFERENCE_CAPTURED', 'QUALITY_SMOKE_PASS') else 2


if __name__ == '__main__':
    raise SystemExit(main())
