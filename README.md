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

真机启动时优先导入独立的 `zelux-device-drivers` 包；包不存在才使用内置回退实现。
诊断接口中的 `hal.driver_source` 会明确报告实际来源。

## robotd 总线 IMU 管理

IMU 页通过 robotctl 和 robotd IPC 管理共享总线型号，Agent 不打开总线。环境变量：ZELUX_ROBOTCTL（默认 /opt/robot/daemon/current/bin/robotctl）、ZELUX_ROBOTD_CONFIG（/etc/robot/robotd.toml）、ZELUX_ROBOTD_SOCKET（/run/robotd.sock）。需要具备新管理能力的配套 robotctl/robotd（API 32），缺能力时显示未知并禁用写入，不因单纯版本号偏差拒绝连接。

命令包括 imu.management_status、imu.model_select、imu.model_restart，以及通过 imu.op 执行的总线 IMU 多姿态机体对齐标定；契约位于 Protocol 1.4.0。Agent 仅订阅 robotd 状态和调用 robotctl，不读取共享串口。标定结果写入 robotd 配置但不立即生效，需在机械支撑、停止动作和现场看护条件下单独重启。配置与运行进程路径不一致、模拟运行、B1 未验证和旧 revision 均拒绝危险操作。只授予运行账号必要的指定配置/服务权限，不开放任意 sudo 命令。

总线 IMU 不使用原 I2C 标定/姿态缓存。设备验收入口见相邻 robotd 仓库 docs/test-plan/imu-unified-acceptance-20261116.md。相关自动回归命令为 `python -m unittest zeluxd.tests.test_robotd_imu_management zeluxd.tests.test_robotd_imu_process -v` 和 `python -m zeluxd.tests.test_imu_mount_calibration`；本轮未运行。
