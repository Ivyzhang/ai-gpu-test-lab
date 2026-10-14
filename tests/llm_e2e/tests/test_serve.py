import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.llm_e2e.serve import model_identity, server_command, require_free_port


class ServeTests(unittest.TestCase):
    def test_weights_and_tokenizer_changes_change_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'config.json').write_text(json.dumps({'architectures': ['Qwen2ForCausalLM'], 'model_type': 'qwen2'}))
            (root / 'model.safetensors').write_bytes(b'fixture weights only')
            (root / 'tokenizer.json').write_text('{}')
            first = model_identity(root)
            (root / 'model.safetensors').write_bytes(b'changed weights')
            second = model_identity(root)
            self.assertNotEqual(first['model_hash'], second['model_hash'])
            (root / 'tokenizer.json').write_text('{"fixture": true}')
            third = model_identity(root)
            self.assertNotEqual(second['model_hash'], third['model_hash'])

    def test_wrong_model_type_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'config.json').write_text(json.dumps({'architectures': ['DistilBertModel']}))
            with self.assertRaises(ValueError):
                model_identity(root)

    def test_missing_weights_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'config.json').write_text(json.dumps({'architectures': ['Qwen2ForCausalLM']}))
            with self.assertRaises(ValueError):
                model_identity(root)

    def test_incompatible_installed_cli_rejected(self):
        with self.assertRaises(ValueError):
            server_command('/usr/bin/trtllm-serve', '/model', '127.0.0.1', 8000, '--host --port')

    def test_constructs_single_gpu_pytorch_backend_command(self):
        cmd = server_command('/bin/trtllm-serve', '/model', '127.0.0.1', 8000,
                             '--host --port --backend pytorch --max_seq_len --max_batch_size')
        self.assertEqual(cmd[:2], ['/bin/trtllm-serve', '/model'])
        self.assertEqual(cmd[cmd.index('--backend') + 1], 'pytorch')
        self.assertEqual(cmd[cmd.index('--max_seq_len') + 1], '4096')

    def test_occupied_port_fails_before_launch(self):
        # Socket binding is an OS boundary; no live port is needed for this unit test.
        with patch('socket.socket') as factory:
            factory.return_value.__enter__.return_value.bind.side_effect = OSError('address already in use')
            with self.assertRaises(RuntimeError):
                require_free_port('127.0.0.1', 8000)


if __name__ == '__main__':
    unittest.main()
