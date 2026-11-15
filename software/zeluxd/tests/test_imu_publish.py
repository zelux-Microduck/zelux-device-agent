from __future__ import annotations

from zeluxd.imu_publish import packet


def test_packet_requires_a_control_frame() -> None:
    assert packet({"ok": True, "roll": 1.0}, 1) is None


def test_packet_preserves_units_and_driver_identity() -> None:
    message = packet(
        {
            "ok": True,
            "gyro_rad_s": [0.1, 0.2, 0.3],
            "gravity": [0.0, 0.0, -1.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "calibrated": True,
            "age_ms": 4,
            "chip": "QMI8658",
        },
        9,
        {"loaded": "zelux.imu.multi-i2c", "version": "1.1.0"},
    )

    assert message is not None
    assert message["sequence"] == 9
    assert message["publisher_id"] == "test-publisher"
    assert message["ready"] is True
    assert message["gyro_rad_s"] == [0.1, 0.2, 0.3]
    assert message["driver"]["id"] == "zelux.imu.multi-i2c"
    assert message["driver"]["model"] == "QMI8658"


def test_control_readiness_does_not_depend_on_studio_calibration_wizard() -> None:
    message = packet(
        {"ok": True, "gyro_rad_s": [0.0, 0.0, 0.0], "gravity": [0.0, 0.0, -1.0],
         "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "age_ms": 5, "calibrated": False},
        1,
    )
    assert message is not None and message["ready"] is True
