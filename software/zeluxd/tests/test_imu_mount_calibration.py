import math
import unittest

from zeluxd.imu_mount_calibration import MountCalibration, solve_correction


def poses(angle_deg=24):
    angle = math.radians(angle_deg)
    s, c = math.sin(angle), math.cos(angle)
    return {
        "still": [0, 0, -1],
        "forward": [s, 0, -c],
        "back": [-s, 0, -c],
        "left": [0, s, -c],
        "right": [0, -s, -c],
    }


class MountCalibrationTests(unittest.TestCase):
    def test_identity_alignment_produces_identity_correction(self):
        result = solve_correction(poses())
        self.assertEqual(result["correction_wxyz"], [1.0, 0.0, 0.0, 0.0])
        self.assertTrue(all(row["ok"] for row in result["checks"]))

    def test_rotated_nominal_frame_is_recovered(self):
        angle = math.radians(24)
        s, c = math.sin(angle), math.cos(angle)
        # Nominal frame is rotated +90° around Z, so input body vectors are Rz(-90°).
        samples = {
            "still": [0, 0, -1],
            "forward": [0, -s, -c],
            "back": [0, s, -c],
            "left": [s, 0, -c],
            "right": [-s, 0, -c],
        }
        result = solve_correction(samples)
        expected = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
        for got, want in zip(result["correction_wxyz"], expected):
            self.assertAlmostEqual(got, want, places=6)

    def test_recalibration_composes_with_existing_correction(self):
        angle = math.radians(24)
        s, c = math.sin(angle), math.cos(angle)
        samples = {
            "still": [0, 0, -1],
            "forward": [0, -s, -c],
            "back": [0, s, -c],
            "left": [s, 0, -c],
            "right": [-s, 0, -c],
        }
        half = math.sqrt(0.5)
        result = solve_correction(samples, [half, 0.0, 0.0, -half])
        for got, want in zip(result["correction_wxyz"], [1.0, 0.0, 0.0, 0.0]):
            self.assertAlmostEqual(got, want, places=6)
        self.assertNotEqual(result["delta_correction_wxyz"], result["correction_wxyz"])

    def test_inconsistent_opposing_tilts_are_rejected(self):
        samples = poses()
        samples["back"] = samples["forward"]
        with self.assertRaisesRegex(ValueError, "相反方向"):
            solve_correction(samples)

    def test_wizard_requires_all_five_captures(self):
        wizard = MountCalibration()
        wizard.start()
        with self.assertRaisesRegex(ValueError, "五个姿态"):
            wizard.solve()
        for key, gravity in poses().items():
            wizard.capture({"gravity": gravity})
        self.assertTrue(wizard.view()["ready"])
        self.assertTrue(wizard.solve()["checks"][0]["ok"])


if __name__ == "__main__":
    unittest.main()
