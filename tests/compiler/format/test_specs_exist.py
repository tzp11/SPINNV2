import unittest
from pathlib import Path


class FormatSpecTests(unittest.TestCase):
    def test_format_specs_exist(self):
        root = Path(__file__).resolve().parents[3]
        self.assertTrue((root / "format" / "sir_spec.md").exists())
        self.assertTrue((root / "format" / "spk_format.md").exists())


if __name__ == "__main__":
    unittest.main()
