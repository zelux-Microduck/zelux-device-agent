# ZELUX Device Agent

运行在机器人主控上的设备服务，负责协议、鉴权、状态聚合、驱动管理和固件管理。
Studio 已拆为独立仓库；Agent 在没有本地 Studio 产物时仍提供 API 和 WebSocket。

```powershell
cd software
python -m pip install -r requirements.txt
$env:ZELUX_MODE='sim'
$env:ZELUX_NO_PASSWORD='1'
python run.py
```

测试：

```powershell
cd software
python -m zeluxd.tests.test_protocol_drivers
python -m zeluxd.tests.test_firmware
```

公共消息契约来自 `zelux-protocol`。具体型号驱动将由 `zelux-device-drivers` 发布，
本仓库只保留迁移期内置实现和 Driver API。
