import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from zeluxd.robotd_imu import RobotdImu
from zeluxd.errors import ZeluxError

class ProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_executable_is_actionable(self):
        bridge = RobotdImu(SimpleNamespace())
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError("missing")):
            with self.assertRaises(ZeluxError):
                await bridge._process("/missing/robotctl", "imu-status")

    async def test_timeout_reaps_child(self):
        process = SimpleNamespace(returncode=None, kill=Mock(),
                                  communicate=AsyncMock(side_effect=[asyncio.TimeoutError(), (b"", b"")]))
        bridge = RobotdImu(SimpleNamespace())
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            with self.assertRaises(ZeluxError):
                await bridge._process("robotctl", "imu-status")
        process.kill.assert_called_once()
        self.assertEqual(process.communicate.await_count, 2)

    async def test_invalid_json_is_not_status(self):
        process = SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(b"not json", b"")))
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            with self.assertRaises(ZeluxError):
                await RobotdImu(SimpleNamespace())._process("robotctl", "imu-status")

