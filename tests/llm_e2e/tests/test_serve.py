import json
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.llm_e2e.serve import model_identity, server_command, require_free_port, main, package_versions


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

    HELP = ('--host --port --max-model-len --max-num-seqs --tensor-parallel-size '
            '--served-model-name --generation-config --dtype --gpu-memory-utilization')

    def test_incompatible_installed_cli_rejected(self):
        with self.assertRaises(ValueError):
            server_command('/bin/vllm', '/model', '127.0.0.1', 8000, '--host --port')

    def test_constructs_single_gpu_vllm_command(self):
        cmd = server_command('/bin/vllm', '/model', '127.0.0.1', 8000, self.HELP)
        self.assertEqual(cmd, [
            '/bin/vllm', 'serve', '/model', '--host', '127.0.0.1', '--port', '8000',
            '--served-model-name', 'Qwen/Qwen2.5-0.5B-Instruct', '--tensor-parallel-size', '1',
            '--max-model-len', '4096', '--max-num-seqs', '8', '--generation-config', 'vllm',
            '--dtype', 'auto', '--gpu-memory-utilization', '0.8',
        ])

    def test_required_option_prefix_does_not_count_as_supported(self):
        help_text = self.HELP.replace('--max-model-len', '--max-model-len-other')
        with self.assertRaises(ValueError):
            server_command('/bin/vllm', '/model', '127.0.0.1', 8000, help_text)

    def test_package_versions_records_vllm(self):
        with patch('tests.llm_e2e.serve.version', side_effect=lambda name: {'vllm': 'fixture-vllm', 'torch': 'fixture-torch', 'transformers': 'fixture-transformers', 'bitsandbytes': 'fixture-bnb', 'accelerate': 'fixture-accelerate'}[name]):
            self.assertEqual(package_versions(), {'vllm': 'fixture-vllm', 'torch': 'fixture-torch', 'transformers': 'fixture-transformers', 'bitsandbytes': 'fixture-bnb', 'accelerate': 'fixture-accelerate'})

    def test_preflight_uses_vllm_subcommand_and_records_runtime_without_launching(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / 'model'
            model.mkdir()
            (model / 'config.json').write_text(json.dumps({'architectures': ['Qwen2ForCausalLM']}))
            (model / 'model.safetensors').write_bytes(b'test fixture only')
            (model / 'tokenizer.json').write_text('{}')
            output = root / 'run'
            commands = []
            def run_command(command, **kwargs):
                commands.append(command)
                if command == ['/bin/vllm', 'serve', '--help=all']:
                    self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-fixture')
                    return subprocess.CompletedProcess(command, 0, self.HELP, '')
                if '--query-gpu=uuid,name,memory.total' in command:
                    return subprocess.CompletedProcess(command, 0, 'GPU-fixture, Fixture GPU, 24000', '')
                if '--query-gpu=driver_version' in command:
                    return subprocess.CompletedProcess(command, 0, 'fixture-driver', '')
                self.fail('unexpected command: ' + str(command))
            with patch('sys.argv', ['serve', '--model-dir', str(model), '--output', str(output), '--cuda-device', 'GPU-fixture', '--preflight-only']), \
                 patch('tests.llm_e2e.serve.require_free_port'), \
                 patch('tests.llm_e2e.serve.shutil.which', side_effect=lambda name: '/bin/' + name), \
                 patch('tests.llm_e2e.serve.subprocess.run', side_effect=run_command), \
                 patch('tests.llm_e2e.serve.package_versions', return_value={'vllm': 'fixture-vllm'}), \
                 patch('tests.llm_e2e.serve.subprocess.Popen', side_effect=AssertionError('preflight must not launch')):
                self.assertEqual(main(), 0)
            manifest = json.loads((output / 'deployment.json').read_text())
            self.assertEqual(manifest['runtime'], 'vllm')
            self.assertEqual(manifest['controlled']['runtime'], 'vllm')
            self.assertEqual(manifest['state'], 'preflight')
            self.assertEqual(manifest['allowed_model_ids'], ['Qwen/Qwen2.5-0.5B-Instruct'])
            self.assertEqual(manifest['versions']['vllm'], 'fixture-vllm')
            self.assertEqual(manifest['command'][:3], ['/bin/vllm', 'serve', str(model.resolve())])
            self.assertEqual(len(commands), 3)

    def test_quantized_loading_requires_both_cli_options(self):
        with self.assertRaises(ValueError):
            server_command('/bin/vllm', '/model', '127.0.0.1', 8000, self.HELP,
                           quantization='bitsandbytes', dtype='half')

    def test_quantized_command_uses_original_weights_and_half_compute(self):
        cmd = server_command('/bin/vllm', '/model', '127.0.0.1', 8000,
                             self.HELP + ' --quantization --load-format',
                             quantization='bitsandbytes', dtype='half')
        self.assertEqual(cmd[cmd.index('--quantization') + 1], 'bitsandbytes')
        self.assertEqual(cmd[cmd.index('--load-format') + 1], 'bitsandbytes')
        self.assertEqual(cmd[cmd.index('--dtype') + 1], 'half')
        self.assertEqual(cmd[2], '/model')

    def test_occupied_port_fails_before_launch(self):
        # Socket binding is an OS boundary; no live port is needed for this unit test.
        with patch('socket.socket') as factory:
            factory.return_value.__enter__.return_value.bind.side_effect = OSError('address already in use')
            with self.assertRaises(RuntimeError):
                require_free_port('127.0.0.1', 8000)


if __name__ == '__main__':
    unittest.main()
