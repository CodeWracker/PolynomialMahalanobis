import hashlib
from pathlib import Path

from tests.helpers import ROOT


def test_polymodel_sha256_unchanged() -> None:
    """
    Goal 00.3 / criterion 16: PolyModel.py must remain byte-a-byte identical
    throughout the migration. No test should modify this file.
    """
    expected = "e31bf63df4c11bd25401153ff1473b95d5504ce3833108fc8ec1c9e4a522ab8e"
    actual = hashlib.sha256((ROOT / "src" / "PolyModel.py").read_bytes()).hexdigest()
    assert actual == expected, (
        f"PolyModel.py SHA256 changed! Expected {expected}, got {actual}"
    )
