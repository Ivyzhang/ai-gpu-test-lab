"""Descriptive statistics and fail-closed absolute/regression gates."""
import math


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def percentile(values, q):
    values = sorted(v for v in values if finite(v))
    return values[max(0, math.ceil(q * len(values)) - 1)] if values else None


def summarize(rows, seconds):
    good = [r for r in rows if r.get('ok')]
    token_rows = [r for r in good if type(r.get('completion_tokens')) is int and r['completion_tokens'] > 0]
    token_complete = len(token_rows) == len(rows) and bool(rows)
    total_tokens = sum(r['completion_tokens'] for r in token_rows)
    result = {'requests': len(rows), 'successes': len(good), 'failures': len(rows) - len(good),
              'success_rate': len(good) / len(rows) if rows else 0,
              'elapsed_s': seconds, 'successful_requests_per_s': len(good) / seconds if seconds > 0 else None,
              'usage_complete': token_complete and all(type(r.get('prompt_tokens')) is int and r['prompt_tokens'] > 0 for r in good),
              'output_tokens_per_s': total_tokens / seconds if token_complete and seconds > 0 else None,
              'successful_output_tokens_per_s': total_tokens / seconds if seconds > 0 else None,
              'one_chunk_responses': sum(r.get('content_chunks') == 1 for r in good)}
    for metric, field in [('ttft', 'ttft_ms'), ('e2e', 'e2e_ms'), ('tpot_estimate', 'tpot_estimate_ms')]:
        values = [r.get(field) for r in good]
        result[metric + '_samples'] = sum(finite(v) for v in values)
        for name, q in [('p50', .5), ('p95', .95), ('p99', .99)]:
            result[metric + '_' + name + '_ms'] = percentile(values, q)
    for field in ('prompt_tokens', 'completion_tokens'):
        result[field + '_p50'] = percentile([r.get(field) for r in good], .5)
        result[field + '_p95'] = percentile([r.get(field) for r in good], .95)
    return result


def evaluate(report, policy, baseline=None, require_baseline=False):
    reasons = []
    cases = report.get('cases', {})
    expected = report.get('expected_cases', [])
    if not report.get('valid'):
        reasons.append('run invalid: inspect infrastructure/preflight errors')
    if not expected or set(expected) != set(cases):
        reasons.append('required case set incomplete')
    if not report.get('functional') or not all(x.get('ok') for x in report['functional']):
        reasons.append('functional checks missing or failed')
    for name, stats in cases.items():
        if stats.get('requests', 0) < policy.get('min_samples', 1):
            reasons.append(name + ': insufficient samples')
        if stats.get('success_rate', 0) < policy.get('min_success_rate', 1):
            reasons.append(name + ': success rate below target')
        if not stats.get('usage_complete'):
            reasons.append(name + ': complete usage required for all measured requests')
        for prefix in ('ttft', 'e2e', 'tpot_estimate'):
            if stats.get(prefix + '_samples', 0) < policy.get('min_samples', 1):
                reasons.append(name + ': insufficient usable ' + prefix + ' samples')
        if stats.get('ttft_samples') != stats.get('successes') or stats.get('e2e_samples') != stats.get('successes'):
            reasons.append(name + ': missing request latency measurements')
        for key in ('ttft_p95_ms', 'e2e_p95_ms', 'tpot_estimate_p95_ms'):
            actual, limit = stats.get(key), policy.get(key)
            if not finite(actual):
                reasons.append(name + ': missing/invalid ' + key)
            elif finite(limit) and actual > limit:
                reasons.append(f'{name}: {key} {actual:.3f} > {limit}')
    if reasons:
        return {'status': 'BLOCKED', 'reasons': reasons}
    if any(not finite(policy.get(k)) or policy[k] <= 0 for k in ('ttft_p95_ms', 'e2e_p95_ms', 'tpot_estimate_p95_ms')):
        return {'status': 'NEEDS_CONFIGURATION', 'reasons': ['set approved, positive SLA limits; example does not invent them']}
    if require_baseline and baseline is None:
        return {'status': 'NOT_COMPARABLE', 'reasons': ['required baseline missing']}
    if baseline is not None:
        if baseline.get('schema_version') != 1 or report.get('schema_version') != 1 or baseline.get('gate', {}).get('status') != 'PASS':
            return {'status': 'NOT_COMPARABLE', 'reasons': ['baseline schema unsupported or baseline gate was not PASS']}
        if not report.get('controlled') or report['controlled'] != baseline.get('controlled') or set(cases) != set(baseline.get('cases', {})):
            return {'status': 'NOT_COMPARABLE', 'reasons': ['controlled conditions or case sets differ']}
        delta = policy.get('max_regression_fraction')
        if not finite(delta):
            return {'status': 'NEEDS_CONFIGURATION', 'reasons': ['invalid max_regression_fraction']}
        for name, stats in cases.items():
            old = baseline['cases'][name]
            if (old.get('requests', 0) < policy.get('min_samples', 1) or not old.get('usage_complete')
                    or not finite(old.get('success_rate')) or not 0 < old['success_rate'] <= 1
                    or old['success_rate'] < policy.get('min_success_rate', 1)
                    or any(old.get(p + '_samples', 0) < policy.get('min_samples', 1) for p in ('ttft', 'e2e', 'tpot_estimate'))):
                return {'status': 'NOT_COMPARABLE', 'reasons': [name + ': baseline incomplete or failed']}
            for key in ('ttft_p95_ms', 'e2e_p95_ms', 'tpot_estimate_p95_ms', 'output_tokens_per_s'):
                before, after = old.get(key), stats.get(key)
                if not finite(before) or before <= 0 or not finite(after):
                    return {'status': 'NOT_COMPARABLE', 'reasons': [name + ': invalid baseline/candidate metric ' + key]}
                regression = (1 - after / before) if key == 'output_tokens_per_s' else (after / before - 1)
                if regression > delta:
                    reasons.append(f'{name}: {key} regressed {regression:.1%} (limit {delta:.1%})')
    return {'status': 'BLOCKED' if reasons else 'PASS', 'reasons': reasons or ['all configured checks passed'],
            'baseline_compared': baseline is not None}
