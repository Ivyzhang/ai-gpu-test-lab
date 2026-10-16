import json
from pathlib import Path
import tempfile
import unittest

from tests.model_deployment.download_models import model_hash
from tests.model_deployment.qwen_quality import cases, correct


class RecipeTests(unittest.TestCase):
    def test_hash_excludes_download_manifest_but_covers_weights_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'model.safetensors').write_bytes(b'fixture only')
            (root/'tokenizer.json').write_text('{}')
            first, _ = model_hash(root)
            (root/'snapshot-manifest.json').write_text(json.dumps({'model_hash': first}))
            self.assertEqual(first, model_hash(root)[0])
            (root/'tokenizer.json').write_text('{"test": 1}')
            second, _ = model_hash(root)
            self.assertNotEqual(first, second)
            (root/'model.safetensors').write_bytes(b'changed fixture')
            self.assertNotEqual(second, model_hash(root)[0])

    def test_missing_weights_cannot_have_valid_model_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                model_hash(Path(tmp))

    def test_answer_oracle_rejects_wrong_value_and_unrequested_prose(self):
        self.assertTrue(correct(' 5\n', '5'))
        self.assertFalse(correct('15', '5'))
        self.assertFalse(correct('The answer is 5', '5'))
        self.assertFalse(correct(None, '5'))

    def test_fixture_tasks_are_distinct_and_have_known_answers(self):
        data = cases()
        self.assertEqual(data[0]['expected'], '5')
        self.assertEqual(data[1]['expected'], '7000')
        self.assertEqual(len({r['id'] for r in data}), 48)


if __name__ == '__main__':
    unittest.main()
