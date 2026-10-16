"""Resolve immutable revisions, download original weights, and record file hashes."""
import argparse
import hashlib
import json
from pathlib import Path


def model_hash(path):
    files = {}
    for p in sorted(Path(path).rglob('*')):
        if not p.is_file() or '.cache' in p.parts or p.name in ('snapshot-manifest.json',):
            continue
        if p.suffix not in ('.json', '.safetensors', '.txt', '.jinja', '.model'):
            continue
        digest = hashlib.sha256()
        with p.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        files[str(p.relative_to(path))] = digest.hexdigest()
    if not any(name.endswith('.safetensors') for name in files):
        raise ValueError('no safetensors weights found')
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/models')
    parser.add_argument('--distilbert-revision', default='main')
    parser.add_argument('--qwen-revision', default='main')
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    for repo, folder, revision in [
        ('distilbert/distilbert-base-uncased', 'distilbert-base-uncased', args.distilbert_revision),
        ('Qwen/Qwen2.5-0.5B-Instruct', 'Qwen2.5-0.5B-Instruct', args.qwen_revision),
    ]:
        destination = Path(args.root) / folder
        if destination.exists():
            raise ValueError(f'{destination} already exists; use a new root to avoid mixing revisions')
        sha = HfApi().model_info(repo_id=repo, revision=revision).sha
        if not sha:
            raise ValueError('no immutable revision returned for ' + repo)
        snapshot_download(repo_id=repo, revision=sha, local_dir=str(destination),
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
        digest, files = model_hash(destination)
        (destination / 'snapshot-manifest.json').write_text(json.dumps(
            {'repo': repo, 'revision': sha, 'model_hash': digest, 'files': files}, indent=2) + '\n')
        print(repo, sha, destination, flush=True)


if __name__ == '__main__':
    main()
