import unittest

from compiler.target.profile import load_target_profile, validate_target_profile


class TargetProfileTests(unittest.TestCase):
    def test_load_cpu_ref_profile(self):
        profile = load_target_profile("cpu_ref")
        self.assertEqual(profile["name"], "cpu_ref")
        self.assertEqual(profile["endianness"], "little")
        self.assertIn("Conv", profile["ops"])
        self.assertIs(profile["memory"]["allow_runtime_malloc"], False)

    def test_validate_rejects_missing_keys(self):
        with self.assertRaisesRegex(ValueError, "missing keys"):
            validate_target_profile({"name": "broken"})


if __name__ == "__main__":
    unittest.main()
