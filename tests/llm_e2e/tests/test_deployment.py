import os
import platform
import unittest
from unittest.mock import patch

from tests.llm_e2e.deployment import verify_live_deployment


class DeploymentTests(unittest.TestCase):
    def manifest(self):
        return {'state': 'ready', 'server_pid': os.getpid(), 'launch_hostname': platform.node(), 'server_port': 8000, 'server_host': '127.0.0.1'}

    def test_ipv6_manifest_cannot_certify_ipv4_endpoint(self):
        m = self.manifest()
        m['server_host'] = '::1'
        with patch('tests.llm_e2e.deployment.listening_pids', return_value={os.getpid()}):
            with self.assertRaises(ValueError):
                verify_live_deployment(m, 'http://127.0.0.1:8000')

    def test_ipv6_endpoint_is_rejected_even_on_matching_port(self):
        with patch('tests.llm_e2e.deployment.listening_pids', return_value={os.getpid()}):
            with self.assertRaises(ValueError):
                verify_live_deployment(self.manifest(), 'http://[::1]:8000')

    def test_preflight_manifest_cannot_certify_an_endpoint(self):
        m = self.manifest()
        m['state'] = 'preflight'
        with self.assertRaises(ValueError):
            verify_live_deployment(m, 'http://127.0.0.1:8000')

    def test_stopped_manifest_cannot_certify_an_endpoint(self):
        m = self.manifest()
        m['state'] = 'stopped'
        with self.assertRaises(ValueError):
            verify_live_deployment(m, 'http://127.0.0.1:8000')

    def test_remote_endpoint_is_rejected(self):
        with self.assertRaises(ValueError):
            verify_live_deployment(self.manifest(), 'http://192.0.2.1:8000')

    def test_live_pid_with_unowned_port_is_rejected(self):
        with patch('tests.llm_e2e.deployment.listening_pids', return_value=set()):
            with self.assertRaises(RuntimeError):
                verify_live_deployment(self.manifest(), 'http://127.0.0.1:8000')

    def test_listener_from_another_process_group_is_rejected(self):
        with patch('tests.llm_e2e.deployment.listening_pids', return_value={987654}), patch('os.getpgid', side_effect=[100, 200]):
            with self.assertRaises(RuntimeError):
                verify_live_deployment(self.manifest(), 'http://127.0.0.1:8000')


if __name__ == '__main__':
    unittest.main()
