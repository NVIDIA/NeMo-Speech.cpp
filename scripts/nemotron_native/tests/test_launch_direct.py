# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import hashlib
import importlib.util
import json
import os
import signal
import socket
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
SPEC = importlib.util.spec_from_file_location(
    "native_launch_direct", Path(__file__).parents[1] / "launch.py"
)
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)


class Fixture:
    def __init__(self, root, direct=True, width=512):
        self.root = root
        self.package = root / 'package'
        self.model = root / 'model'
        self.model.mkdir()
        self.tokenizer = self.model / 'tokenizer.json'
        self.tokenizer.write_text('{"fixture":true}')
        self.library = root / 'libpreparation.so'
        self.library.touch()
        self.serving = self.package / 'programs' / ('b' + str(width)) / 'serving.json'
        self.serving.parent.mkdir(parents=True)
        self.plan = {
            'physical_width': width,
            'host': {'batching_mode': 'inflight'},
            'model': {'checkpoint_sha256': 'a' * 64},
            'transport_compatibility': {
                'output_contract': 'pcm_f32_le_22050_mono_variable_streaming_v3',
                'installed_transport_versions': {
                    'trt_version': 'not-installed',
                    'trtllm_version': 'not-installed',
                    'trtllm_commit': 'not-used',
                },
            },
        }
        self.save_plan()
        self.seal = {
            'model_identity': 'a' * 64,
            'files': {
                'tokenizer.json': {
                    'bytes': self.tokenizer.stat().st_size,
                    'sha256': hashlib.sha256(self.tokenizer.read_bytes()).hexdigest(),
                }
            },
        }
        self.seal_path = self.model / 'model-seal.json'
        self.save_seal()
        (self.package / 'PACKAGE.json').write_text('{}')
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        self.arguments = [
            '--package',
            str(self.package),
            '--model',
            str(self.model),
            '--width',
            str(width),
            '--listen',
            str(root / 'bridge.sock'),
            '--receipts',
            str(root / 'receipts'),
            '--http-port',
            str(port),
            '--frontend-binary',
            '/test/frontend',
            '--startup-timeout',
            '3',
            '--shutdown-timeout',
            '0.05',
        ]
        if direct:
            self.arguments.append('--direct-worker')
        self.args = launch.parse_args(self.arguments)

    def save_plan(self):
        self.serving.write_text(json.dumps(self.plan))

    def save_seal(self):
        self.seal_path.write_text(json.dumps(self.seal))

    def configuration(self):
        with mock.patch.dict(os.environ, NEMOTRON_NATIVE_PREPARATION_LIBRARY=str(self.library)):
            return launch.direct_worker_configuration(self.args)


class MetadataTests(unittest.TestCase):
    def test_default_path_is_unchanged_and_direct_requires_http(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), direct=False)
            self.assertFalse(fixture.args.direct_worker)
            args = fixture.arguments[:]
            port = args.index('--http-port')
            del args[port : port + 2]
            with self.assertRaises(SystemExit):
                launch.parse_args(args + ['--direct-worker'])
            worker, bridge = launch.commands(fixture.args)
            self.assertIn('--qualification-only', worker)
            self.assertIn('nemotron_cpp_bridge', bridge)
            self.assertNotIn('DIRECT_WORKER', str(worker))

    def test_metadata_binds_exact_manifest_model_tokenizer_width_versions_and_normalization(self):
        for width in (1, 32, 512, 4096):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary), width=width)
                config = fixture.configuration()
                self.assertEqual(
                    config['manifest_sha256'],
                    hashlib.sha256(fixture.serving.read_bytes()).hexdigest(),
                )
                self.assertEqual(config['model_sha256'], 'a' * 64)
                self.assertEqual(config['enabled_buckets'], sorted({1, width}))
                self.assertEqual(
                    config['installed_transport_versions'],
                    fixture.plan['transport_compatibility']['installed_transport_versions'],
                )
                self.assertEqual(config['mode'], 'inflight')
                self.assertEqual(config['text_preparation']['tokenizer'], str(fixture.tokenizer))
                self.assertEqual(config['text_preparation']['library'], str(fixture.library))
                self.assertFalse(config['text_normalization_enabled'])
                normalization = fixture.root / 'normalization.json'
                normalization.write_text('{"enabled":true,"workers":16}')
                fixture.args.normalization_config = normalization
                config = fixture.configuration()
                self.assertTrue(config['text_normalization_enabled'])
                self.assertEqual(
                    json.loads(config['text_preparation']['normalization_config']),
                    {'enabled': True, 'workers': 16},
                )

    def test_width_mode_output_and_model_identity_rejected(self):
        changes = [
            ('physical_width', 32),
            ('physical_width', True),
            ('host', {'batching_mode': 'fixed_cohort'}),
            ('model', {'checkpoint_sha256': 'invalid'}),
            ('model', {'checkpoint_sha256': 123}),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                fixture.plan[key] = value
                fixture.save_plan()
                with self.assertRaises(ValueError):
                    fixture.configuration()
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fixture.plan['transport_compatibility']['output_contract'] = 'pcm16'
            fixture.save_plan()
            with self.assertRaisesRegex(ValueError, 'streaming-v3'):
                fixture.configuration()

    def test_malformed_transport_versions_rejected_before_launch(self):
        bad_versions = [
            None,
            [],
            {},
            {'trt_version': 'x'},
            {'trt_version': True, 'trtllm_version': 'x', 'trtllm_commit': 'x'},
            {'trt_version': '', 'trtllm_version': 'x', 'trtllm_commit': 'x'},
            {'trt_version': 'x' * 1025, 'trtllm_version': 'x', 'trtllm_commit': 'x'},
        ]
        for value in bad_versions:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                fixture.plan['transport_compatibility']['installed_transport_versions'] = value
                fixture.save_plan()
                with self.assertRaisesRegex(ValueError, 'transport versions'):
                    fixture.configuration()

    def test_wrong_tokenizer_bytes_hash_model_and_escape_rejected(self):
        for change in ('size', 'hash', 'model', 'escape'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                if change == 'size':
                    fixture.seal['files']['tokenizer.json']['bytes'] += 1
                elif change == 'hash':
                    fixture.seal['files']['tokenizer.json']['sha256'] = 'b' * 64
                elif change == 'model':
                    fixture.seal['model_identity'] = 'b' * 64
                else:
                    escaped = fixture.root / 'escape.json'
                    escaped.write_bytes(fixture.tokenizer.read_bytes())
                    fixture.tokenizer.unlink()
                    fixture.tokenizer.symlink_to(escaped)
                fixture.save_seal()
                with self.assertRaisesRegex(ValueError, 'sealed model'):
                    fixture.configuration()

    def test_library_and_normalizer_configuration_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            with mock.patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
                ValueError, 'preparation library'
            ):
                launch.direct_worker_configuration(fixture.args)
            fixture.library.unlink()
            with self.assertRaises(FileNotFoundError):
                fixture.configuration()
            fixture.library.touch()
            config = fixture.root / 'normalization.json'
            fixture.args.normalization_config = config
            for invalid in ('[]', '{"enabled":1}'):
                config.write_text(invalid)
                with self.assertRaises(ValueError):
                    fixture.configuration()


class DirectLaunchTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_children_only_frontend_gets_config_and_requested_affinity(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            available = sorted(os.sched_getaffinity(0))
            fixture.args.native_cpus = str(available[0])
            fixture.args.frontend_cpus = ','.join(map(str, available[:2]))
            worker_observed = fixture.root / 'worker-observed.json'
            frontend_observed = fixture.root / 'frontend-observed.json'
            ready = fixture.args.receipts / 'worker/NATIVE_READY.json'
            worker_code = f'''import json,os,pathlib,signal,time
signal.signal(signal.SIGTERM,signal.SIG_IGN)
p=pathlib.Path({str(worker_observed)!r})
p.write_text(json.dumps({{'config':os.getenv('NEMOTRON_DIRECT_WORKER_CONFIG'),'cpus':sorted(os.sched_getaffinity(0))}}))
p=pathlib.Path({str(ready)!r});p.parent.mkdir(parents=True);p.write_text('{{}}')
time.sleep(60)
'''
            frontend_code = f'''import json,os,pathlib,socket,time
p=pathlib.Path({str(frontend_observed)!r})
p.write_text(json.dumps({{'config':os.getenv('NEMOTRON_DIRECT_WORKER_CONFIG'),'cpus':sorted(os.sched_getaffinity(0))}}))
s=socket.socket();s.bind(('127.0.0.1',{fixture.args.http_port}));s.listen(1)
time.sleep(60)
'''
            original_spawn = asyncio.create_subprocess_exec
            spawned = []

            async def spawn(*command, **kwargs):
                spawned.append((command, kwargs['env'].copy()))
                if '/test/frontend' in command:
                    index = command.index('/test/frontend')
                    command = (*command[:index], sys.executable, '-c', frontend_code)
                return await original_spawn(*command, **kwargs)

            callbacks = {}
            loop = asyncio.get_running_loop()
            with mock.patch.dict(
                os.environ,
                NEMOTRON_NATIVE_PREPARATION_LIBRARY=str(fixture.library),
                NEMOTRON_DIRECT_WORKER_CONFIG='/do/not/inherit.json',
            ), mock.patch.object(
                launch,
                'commands',
                return_value=([sys.executable, '-c', worker_code], ['/must/not/spawn/bridge']),
            ), mock.patch.object(
                launch.asyncio, 'create_subprocess_exec', side_effect=spawn
            ), mock.patch.object(
                loop,
                'add_signal_handler',
                side_effect=lambda number, callback: callbacks.setdefault(number, callback),
            ):
                task = asyncio.create_task(launch.run(fixture.args))
                try:

                    async def observe():
                        while not (fixture.args.receipts / 'LAUNCH.json').exists():
                            if task.done():
                                task.result()
                            await asyncio.sleep(0.01)

                    await asyncio.wait_for(observe(), 3)
                finally:
                    callbacks[signal.SIGTERM]()
                    await asyncio.wait_for(task, 3)
            self.assertEqual(len(spawned), 2)
            self.assertNotIn('NEMOTRON_DIRECT_WORKER_CONFIG', spawned[0][1])
            config_path = fixture.args.receipts / 'DIRECT_WORKER.json'
            self.assertEqual(spawned[1][1]['NEMOTRON_DIRECT_WORKER_CONFIG'], str(config_path))
            self.assertEqual(stat.S_IMODE(config_path.stat().st_mode), 0o600)
            self.assertEqual(json.loads(config_path.read_text()), fixture.configuration())
            self.assertIn(str(fixture.args.native_worker_socket), spawned[1][0])
            self.assertNotIn(str(fixture.args.listen), spawned[1][0])
            self.assertFalse((fixture.args.receipts / 'bridge.log').exists())
            self.assertFalse(fixture.args.listen.exists())
            self.assertEqual(
                json.loads(worker_observed.read_text()), {'config': None, 'cpus': available[:1]}
            )
            self.assertEqual(
                json.loads(frontend_observed.read_text()),
                {'config': str(config_path), 'cpus': available[:2]},
            )
            report = json.loads((fixture.args.receipts / 'EXIT.json').read_text())
            self.assertEqual(report['transport'], 'direct_worker')
            self.assertEqual(report['status'], 'STOPPED')
            self.assertEqual(report['exit_codes'], [-signal.SIGKILL, -signal.SIGTERM])
            self.assertEqual(set(report['cpu_affinity']['roles']), {'worker', 'frontend'})

    async def test_bad_metadata_no_spawn_and_socket_ownership_preserved(self):
        for error in ('metadata', 'socket', 'receipts'):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                if error == 'metadata':
                    fixture.plan['transport_compatibility']['installed_transport_versions'] = {}
                    fixture.save_plan()
                elif error == 'socket':
                    fixture.args.native_worker_socket.write_text('owned')
                else:
                    fixture.args.receipts.mkdir()
                    (fixture.args.receipts / 'owned').write_text('owned')
                with mock.patch.dict(
                    os.environ, NEMOTRON_NATIVE_PREPARATION_LIBRARY=str(fixture.library)
                ), mock.patch.object(
                    launch.asyncio, 'create_subprocess_exec'
                ) as spawn, self.assertRaises(
                    ValueError
                ):
                    await launch.run(fixture.args)
                spawn.assert_not_called()
                if error == 'socket':
                    self.assertEqual(fixture.args.native_worker_socket.read_text(), 'owned')


if __name__ == "__main__":
    unittest.main()
