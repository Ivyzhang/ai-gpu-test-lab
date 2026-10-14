"""Local process/socket attribution: do not certify a server using another process's manifest."""
import os
from pathlib import Path
import platform
import subprocess
from urllib.parse import urlsplit


def listening_pids(port):
    if platform.system() == 'Linux':
        inodes = set()
        for name in ('tcp',):
            path = Path('/proc/net') / name
            if not path.exists():
                continue
            for line in path.read_text().splitlines()[1:]:
                fields = line.split()
                address, port_hex = fields[1].split(':')
                if address in ('0100007F', '00000000') and int(port_hex, 16) == port and fields[3] == '0A':
                    inodes.add('socket:[' + fields[9] + ']')
        owners = set()
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                for fd in (proc / 'fd').iterdir():
                    try:
                        if os.readlink(fd) in inodes:
                            owners.add(int(proc.name))
                    except OSError:
                        continue
            except OSError:
                continue
        return owners
    if platform.system() == 'Darwin':
        result = subprocess.run(['/usr/sbin/lsof', '-nP', '-t', f'-i4TCP@127.0.0.1:{port}', '-sTCP:LISTEN'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode not in (0, 1):
            raise RuntimeError('cannot inspect listening socket ownership')
        return {int(x) for x in result.stdout.split() if x.isdigit()}
    raise RuntimeError('socket ownership verification supports Linux and macOS only')


def verify_live_deployment(manifest, base_url):
    parsed = urlsplit(base_url)
    if manifest.get('state') != 'ready':
        raise ValueError('deployment manifest is not ready (preflight, stopped, or failed)')
    if manifest.get('launch_hostname') != platform.node():
        raise ValueError('run the client on the server host/in the same container for process attribution')
    if (manifest.get('server_host') != '127.0.0.1' or parsed.hostname not in ('127.0.0.1', 'localhost')
            or parsed.scheme != 'http' or parsed.path.rstrip('/')):
        raise ValueError('this suite requires the launched 127.0.0.1 HTTP endpoint; remote/IPv6/gateway attribution is not implemented')
    port = parsed.port or 80
    if port != manifest.get('server_port'):
        raise ValueError('base_url port does not match deployment')
    pid = manifest.get('server_pid')
    if type(pid) is not int or pid < 1:
        raise ValueError('deployment missing server PID')
    try:
        os.kill(pid, 0)
        group = os.getpgid(pid)
        owners = listening_pids(port)
        matched = []
        for owner in owners:
            try:
                if os.getpgid(owner) == group:
                    matched.append(owner)
            except ProcessLookupError:
                continue
    except OSError as exc:
        raise RuntimeError('launched server process is unavailable') from exc
    if not matched:
        raise RuntimeError('endpoint is not owned by the launched server process group')
    return {'server_pid': pid, 'listener_pids': sorted(matched), 'server_port': port}
