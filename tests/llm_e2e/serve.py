"""Preflight and launch a local Qwen2 vLLM service; no downloads/installations."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

from .client import Client
from .deployment import verify_live_deployment
from .run import digest, save, wait_ready


def require_free_port(host, port):
    try:
        family = socket.AF_INET6 if ':' in host else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
    except OSError as exc:
        raise RuntimeError('service port unavailable; stop the old service or choose another port') from exc


def model_identity(root):
    root = Path(root).resolve()
    config = json.loads((root / 'config.json').read_text())
    if config.get('architectures') != ['Qwen2ForCausalLM']:
        raise ValueError('expected Qwen2ForCausalLM; use the complete Qwen2.5-0.5B-Instruct snapshot')
    weights = sorted(root.glob('*.safetensors'))
    if not weights or not (root / 'tokenizer.json').exists():
        raise ValueError('missing local safetensors weights or tokenizer.json')
    paths = sorted(set(weights + list(root.glob('*.json')) + list(root.glob('*.txt')) + list(root.glob('*.jinja'))))
    hashes = {}
    for path in paths:
        hasher = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                hasher.update(chunk)
        hashes[path.name] = hasher.hexdigest()
    index_path = root / 'model.safetensors.index.json'
    if index_path.exists():
        required = set(json.loads(index_path.read_text())['weight_map'].values())
        if not required <= set(hashes):
            raise ValueError('incomplete sharded model snapshot')
    return {'model_hash': digest(hashes), 'file_hashes': hashes, 'model_config': config}


MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'


def server_command(executable, model, host, port, help_text, quantization="none", dtype="auto"):
    if quantization not in ("none", "bitsandbytes") or dtype not in ("auto", "half", "bfloat16"):
        raise ValueError("unsupported quantization or dtype")
    required = ('--host', '--port', '--max-model-len', '--max-num-seqs',
                '--tensor-parallel-size', '--served-model-name', '--generation-config',
                '--dtype', '--gpu-memory-utilization')
    if quantization == 'bitsandbytes':
        required += ('--quantization', '--load-format')
    available = set(re.findall(r'--[a-zA-Z][a-zA-Z0-9_-]*', help_text))
    missing = [flag for flag in required if flag not in available]
    if missing:
        raise ValueError('installed vllm serve CLI incompatible; inspect saved cli-help.txt. Missing: ' + ', '.join(missing))
    command = [executable, 'serve', str(model), '--host', host, '--port', str(port),
            '--served-model-name', MODEL_ID, '--tensor-parallel-size', '1',
            '--max-model-len', '4096', '--max-num-seqs', '8',
            '--generation-config', 'vllm', '--dtype', dtype, '--gpu-memory-utilization', '0.8']
    if quantization == 'bitsandbytes':
        command += ['--quantization', 'bitsandbytes', '--load-format', 'bitsandbytes']
    return command


def package_versions():
    result = {}
    for name in ('vllm', 'torch', 'transformers', 'bitsandbytes', 'accelerate'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = 'not-installed-as-distribution'
    return result


def telemetry(stop, output):
    query = 'timestamp,index,uuid,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu'
    with (output / 'gpu.csv').open('w', encoding='utf-8') as handle:
        handle.write(query + '\n')
        while not stop.is_set():
            try:
                value = subprocess.run(['nvidia-smi', '--query-gpu=' + query, '--format=csv,noheader,nounits'],
                                       capture_output=True, text=True, timeout=5, check=True)
                handle.write(value.stdout)
                handle.flush()
            except Exception as exc:
                with (output / 'telemetry-errors.log').open('a', encoding='utf-8') as errors:
                    errors.write(f'{datetime.now(timezone.utc).isoformat()} {exc}\n')
            stop.wait(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, help='local complete Qwen2.5-0.5B-Instruct snapshot')
    parser.add_argument('--output', required=True, help='new server-log directory')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--cuda-device', default='0', help='one physical CUDA device index or GPU UUID')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--quantization', choices=('none', 'bitsandbytes'), default='none', help='bitsandbytes uses runtime 4-bit loading of original weights')
    parser.add_argument('--dtype', choices=('auto', 'half', 'bfloat16'), default='auto')
    parser.add_argument('--startup-timeout-s', type=float, default=300)
    args = parser.parse_args()
    output = Path(args.output)
    child = None
    stop = threading.Event()
    monitor = None
    previous_handlers = {}
    manifest = None
    created_output = False
    try:
        output.mkdir(parents=True, exist_ok=False)
        created_output = True
        if not 1 <= args.port <= 65535 or ',' in args.cuda_device or not args.cuda_device:
            raise ValueError('valid port and exactly one CUDA device required')
        if args.host != '127.0.0.1' or not 0 < args.startup_timeout_s < 86400:
            raise ValueError('this launcher requires host 127.0.0.1 and a finite positive startup timeout')
        require_free_port(args.host, args.port)
        executable = shutil.which('vllm')
        if not executable or not shutil.which('nvidia-smi'):
            raise RuntimeError('requires an NVIDIA GPU environment with vllm and nvidia-smi installed')
        model = Path(args.model_dir).resolve()
        identity = model_identity(model)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.cuda_device)
        help_result = subprocess.run([executable, 'serve', '--help=all'], capture_output=True, text=True, timeout=60, env=env)
        help_text = help_result.stdout + help_result.stderr
        (output / 'cli-help.txt').write_text(help_text)
        if help_result.returncode:
            raise RuntimeError('vllm serve --help=all failed; inspect cli-help.txt')
        command = server_command(executable, model, args.host, args.port, help_text, args.quantization, args.dtype)
        gpu = subprocess.run(['nvidia-smi', '-i', args.cuda_device,
                              '--query-gpu=uuid,name,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=15, check=True).stdout.strip()
        driver = subprocess.run(['nvidia-smi', '-i', args.cuda_device, '--query-gpu=driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=15, check=True).stdout.strip()
        manifest = {'schema_version': 1, 'state': 'preflight', 'launch_hostname': platform.node(), 'server_port': args.port,
                    'server_host': args.host,
                    'created_utc': datetime.now(timezone.utc).isoformat(),
                    'runtime': 'vllm',
                    'requested_model': MODEL_ID,
                    'allowed_model_ids': [MODEL_ID],
                    'versions': dict(package_versions(), python=sys.version, driver=driver),
                    'identity': identity,
                    'controlled': {'runtime': 'vllm', 'quantization': args.quantization, 'dtype': args.dtype, 'model_hash': identity['model_hash'], 'gpu': gpu,
                                   'server_command': command[1:], 'cuda_visible_devices': args.cuda_device,
                                   'cache_policy': 'installed server defaults; warm shared-prefix workload; not cold-cache benchmark'},
                    'command': command,
                    'validation': 'CLI/model-file preflight only; successful model execution requires the HTTP suite'}
        save(output / 'deployment.json', manifest)
        print('Launch command:', json.dumps(command), flush=True)
        if args.preflight_only:
            return 0
        require_free_port(args.host, args.port)
        with (output / 'server.log').open('w', encoding='utf-8') as log:
            started = time.perf_counter()
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            manifest.update(state='starting', server_pid=child.pid)
            save(output / 'deployment.json', manifest)
            def forward(signum, _frame):
                if child.poll() is None:
                    os.killpg(child.pid, signum)
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[sig] = signal.signal(sig, forward)
            monitor = threading.Thread(target=telemetry, args=(stop, output), daemon=True)
            monitor.start()
            print(f'Server PID {child.pid}; waiting for readiness; logs: {output / "server.log"}', flush=True)
            url = f'http://{args.host}:{args.port}'
            client = Client(url, timeout_s=min(30, args.startup_timeout_s))
            actual_model = wait_ready(client, None, args.startup_timeout_s, process_alive=lambda: child.poll() is None)
            if actual_model not in manifest['allowed_model_ids']:
                raise RuntimeError('server exposed an unexpected model ID: ' + actual_model)
            # Mark ready in memory for attribution; publish only after a real generation succeeds.
            manifest['state'] = 'ready'
            manifest['attribution'] = verify_live_deployment(manifest, url)
            probe = client.generate(actual_model, 'Say hello.', max_tokens=8)
            save(output / 'startup-probe.json', probe)
            if not probe['ok'] or child.poll() is not None:
                raise RuntimeError('real generation readiness probe failed; inspect startup-probe.json and server.log')
            manifest['startup_to_first_generation_ms'] = (time.perf_counter() - started) * 1000
            manifest['validation'] = 'local process/socket attribution and one real generation passed; full suite still required'
            save(output / 'deployment.json', manifest)
            print('READY: run the client in another terminal in this same host/container.', flush=True)
            code = child.wait()
        save(output / 'server-exit.json', {'returncode': code, 'ended_utc': datetime.now(timezone.utc).isoformat()})
        return 0 if code == 0 else 2
    except Exception as exc:
        if created_output:
            error_path = output / ('preflight-error-' + str(time.time_ns()) + '.txt')
            error_path.write_text(f'{type(exc).__name__}: {exc}\n')
        if manifest and child:
            manifest.update(state='failed', error=str(exc))
            save(output / 'deployment.json', manifest)
        print(str(exc), file=sys.stderr)
        return 2
    finally:
        stop.set()
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if monitor:
            monitor.join(timeout=6)
        if manifest and child:
            if manifest['state'] != 'failed':
                manifest['state'] = 'stopped'
            save(output / 'deployment.json', manifest)
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)


if __name__ == '__main__':
    sys.exit(main())
