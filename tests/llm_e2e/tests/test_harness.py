import json
import math
import threading
import time
import unittest

from tests.llm_e2e.client import StreamAccumulator
from tests.llm_e2e.metrics import evaluate, summarize
from tests.llm_e2e.run import parallel_requests


def event(content=None, role=None, finish=None, usage=None):
    delta = {}
    if content is not None:
        delta['content'] = content
    if role:
        delta['role'] = role
    value = {'id': 'completion-1', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
    if usage is not None:
        value['choices'] = []
        value['usage'] = usage
    return json.dumps(value)


class StreamTests(unittest.TestCase):
    def test_role_is_not_first_token_and_chunk_is_not_token(self):
        s = StreamAccumulator(10.0)
        s.feed(event(role='assistant'), 10.1)
        s.feed(event(content='hello world'), 10.3)
        s.feed(event(content='!'), 10.7)
        s.feed(event(finish='stop'), 10.8)
        s.feed(event(usage={'prompt_tokens': 9, 'completion_tokens': 3}), 10.85)
        s.feed('[DONE]', 10.9)
        r = s.result()
        self.assertTrue(r['ok'])
        self.assertAlmostEqual(r['ttft_ms'], 300)
        self.assertAlmostEqual(r['e2e_ms'], 900)
        self.assertAlmostEqual(r['tpot_estimate_ms'], 200)
        self.assertEqual(r['content_chunks'], 2)
        self.assertEqual(r['completion_tokens'], 3)
        self.assertEqual(r['text'], 'hello world!')

    def test_missing_done_is_failure_even_after_finish(self):
        s = StreamAccumulator(0)
        s.feed(event(content='hello'), 0.1)
        s.feed(event(finish='stop'), 0.2)
        self.assertFalse(s.result()['ok'])

    def test_missing_usage_does_not_invent_tokens(self):
        s = StreamAccumulator(0)
        s.feed(event(content='many tokens in one chunk'), 0.1)
        s.feed(event(finish='stop'), 0.2)
        s.feed('[DONE]', 0.3)
        r = s.result()
        self.assertTrue(r['ok'])
        self.assertIsNone(r['completion_tokens'])
        self.assertIsNone(r['tpot_estimate_ms'])

    def test_one_token_has_no_tpot(self):
        s = StreamAccumulator(0)
        s.feed(event(content='2'), 0.1)
        s.feed(event(finish='stop'), 0.2)
        s.feed(event(usage={'prompt_tokens': 5, 'completion_tokens': 1}), 0.3)
        s.feed('[DONE]', 0.4)
        self.assertIsNone(s.result()['tpot_estimate_ms'])

    def test_multiple_tokens_in_one_chunk_have_no_measurable_tpot(self):
        s = StreamAccumulator(0)
        s.feed(event(content='hello world'), 0.1)
        s.feed(event(finish='stop'), 0.2)
        s.feed(event(usage={'prompt_tokens': 5, 'completion_tokens': 2}), 0.3)
        s.feed('[DONE]', 0.4)
        self.assertIsNone(s.result()['tpot_estimate_ms'])

    def test_empty_response_fails(self):
        s = StreamAccumulator(0)
        s.feed(event(finish='stop'), 0.1)
        s.feed('[DONE]', 0.2)
        self.assertFalse(s.result()['ok'])

    def test_stream_error_is_not_success(self):
        s = StreamAccumulator(0)
        with self.assertRaises(ValueError):
            s.feed(json.dumps({'error': {'message': 'engine failed'}}), 0.1)

    def test_response_id_switch_rejected(self):
        s = StreamAccumulator(0)
        s.feed(event(content='a'), 0.1)
        value = json.loads(event(content='b'))
        value['id'] = 'other-request'
        with self.assertRaises(ValueError):
            s.feed(json.dumps(value), 0.2)


def record(ok=True):
    return {'ok': ok, 'ttft_ms': 100 if ok else None,
            'e2e_ms': 300, 'tpot_estimate_ms': 50 if ok else None,
            'completion_tokens': 5 if ok else None, 'prompt_tokens': 10 if ok else None}


def policy():
    return {'min_samples': 2, 'min_success_rate': 1.0,
            'ttft_p95_ms': 200, 'e2e_p95_ms': 500,
            'tpot_estimate_p95_ms': 80, 'max_regression_fraction': 0.10}


def report():
    return {'schema_version': 1, 'gate': {'status': 'PASS'}, 'functional': [{'name': 'smoke', 'ok': True}],
            'expected_cases': ['short-c1'], 'cases': {'short-c1': summarize([record(), record()], 1)},
            'controlled': {'model_hash': 'abc', 'gpu': 'gpu1'}, 'valid': True}


class GateTests(unittest.TestCase):
    def test_failure_is_in_success_rate_denominator(self):
        s = summarize([record(), record(False)], 2)
        self.assertEqual(s['success_rate'], 0.5)
        self.assertEqual(s['output_tokens_per_s'], None)
        self.assertEqual(s['successful_output_tokens_per_s'], 2.5)

    def test_no_samples_fails_closed(self):
        r = report()
        r['cases']['short-c1'] = summarize([], 1)
        self.assertNotEqual(evaluate(r, policy())['status'], 'PASS')

    def test_missing_sla_never_passes(self):
        p = policy()
        p['ttft_p95_ms'] = None
        self.assertEqual(evaluate(report(), p)['status'], 'NEEDS_CONFIGURATION')

    def test_complete_measurement_can_pass_absolute_sla(self):
        self.assertEqual(evaluate(report(), policy())['status'], 'PASS')

    def test_required_baseline_missing_is_not_comparable(self):
        self.assertEqual(evaluate(report(), policy(), require_baseline=True)['status'], 'NOT_COMPARABLE')

    def test_environment_mismatch_is_not_comparable(self):
        baseline = report()
        baseline['controlled']['gpu'] = 'other'
        self.assertEqual(evaluate(report(), policy(), baseline)['status'], 'NOT_COMPARABLE')

    def test_tail_regression_blocks(self):
        candidate = report()
        candidate['cases']['short-c1']['ttft_p95_ms'] = 150
        self.assertEqual(evaluate(candidate, policy(), report())['status'], 'BLOCKED')

    def test_missing_expected_case_blocks(self):
        r = report()
        r['expected_cases'].append('mixed-c8')
        self.assertEqual(evaluate(r, policy())['status'], 'BLOCKED')

    def test_missing_usage_blocks(self):
        r = report()
        rows = [record(), record()]
        rows[0]['completion_tokens'] = None
        r['cases']['short-c1'] = summarize(rows, 1)
        self.assertEqual(evaluate(r, policy())['status'], 'BLOCKED')

    def test_nan_sla_is_configuration_error(self):
        p = policy()
        p['ttft_p95_ms'] = math.nan
        self.assertEqual(evaluate(report(), p)['status'], 'NEEDS_CONFIGURATION')

    def test_failed_baseline_cannot_authorize_release(self):
        baseline = report()
        baseline['gate'] = {'status': 'BLOCKED'}
        self.assertEqual(evaluate(report(), policy(), baseline)['status'], 'NOT_COMPARABLE')

    def test_baseline_requires_matching_schema(self):
        baseline = report()
        baseline['schema_version'] = 999
        self.assertEqual(evaluate(report(), policy(), baseline)['status'], 'NOT_COMPARABLE')

    def test_one_missing_request_latency_cannot_hide_in_percentile(self):
        rows = [record(), record()]
        rows[0]['ttft_ms'] = None
        r = report()
        r['cases']['short-c1'] = summarize(rows, 1)
        self.assertEqual(evaluate(r, policy())['status'], 'BLOCKED')

    def test_nan_baseline_success_rate_is_not_comparable(self):
        baseline = report()
        baseline['cases']['short-c1']['success_rate'] = math.nan
        self.assertEqual(evaluate(report(), policy(), baseline)['status'], 'NOT_COMPARABLE')

    def test_tpot_requires_its_own_minimum_samples(self):
        rows = [record(), record()]
        rows[0].update(completion_tokens=1, tpot_estimate_ms=None)
        r = report()
        r['cases']['short-c1'] = summarize(rows, 1)
        self.assertEqual(evaluate(r, policy())['status'], 'BLOCKED')


class ParallelTests(unittest.TestCase):
    def test_requests_really_overlap_and_respect_limit(self):
        lock = threading.Lock()
        active, peak = 0, 0
        def call(index):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.025)
            with lock:
                active -= 1
            return {'index': index}
        rows, elapsed = parallel_requests(call, count=8, concurrency=4)
        self.assertEqual(sorted(r['index'] for r in rows), list(range(8)))
        self.assertGreater(peak, 1)
        self.assertLessEqual(peak, 4)
        self.assertGreater(elapsed, 0)

    def test_worker_exception_kept_as_failure(self):
        def call(index):
            raise RuntimeError('simulated transport failure')
        rows, _ = parallel_requests(call, count=3, concurrency=2)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(not r['ok'] for r in rows))


if __name__ == '__main__':
    unittest.main()
