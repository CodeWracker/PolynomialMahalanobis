"""Verify no legacy LUT/band-inversion code remains in src/main.py.

Also verifies:
- PolyModel.py SHA256 against golden master baseline
- requirements.txt contains required dependencies
- README.md documents new CLI arguments (cache, audit, hash)
"""

from pathlib import Path

MAIN_PY = Path(__file__).parent.parent / "src" / "main.py"

_LEGACY_PATTERNS: list[str] = [
    "_LUT_VAL",
    "_LUT_DONE",
    "_LUT_LEN",
    "memmap",
    "[..., ::-1]",
    "<< 16",
    "<< 8",
    "codes_flat",
    "missing_codes",
    ".lut.val",
    ".lut.done",
]


def test_no_legacy_code_in_main() -> None:
    """Assert src/main.py contains no legacy LUT/band-inversion patterns."""
    source: str = MAIN_PY.read_text(encoding="utf-8")
    found: list[str] = []
    for pattern in _LEGACY_PATTERNS:
        if pattern in source:
            found.append(pattern)
    assert not found, (
        f"Legacy patterns found in src/main.py: {', '.join(found)}"
    )


def test_polymodel_file_unchanged() -> None:
    """
    PolyModel.py must not have been altered by any goal.
    Compare SHA256 hash of the final file against the registered baseline.
    """
    import hashlib

    poly_model_path: Path = Path(__file__).parent.parent / "src" / "PolyModel.py"
    current_hash: str = hashlib.sha256(poly_model_path.read_bytes()).hexdigest()

    baseline_file: Path = (
        Path(__file__).parent.parent
        / "tests"
        / "golden_master"
        / "PolyModel.py.sha256"
    )
    if not baseline_file.exists():
        import pytest
        pytest.skip(
            "Baseline de hash do PolyModel.py não existe. "
            "Rode 'sha256sum src/PolyModel.py > tests/golden_master/PolyModel.py.sha256' "
            "no início do projeto, antes do Goal 01."
        )
    baseline_hash: str = baseline_file.read_text().split()[0]
    assert current_hash == baseline_hash, (
        "PolyModel.py foi alterado! Isso viola a regra inegociável do projeto: "
        "PolyModel.py não deve ser tocado em nenhum objetivo."
    )


def test_requirements_has_new_dependencies() -> None:
    """requirements.txt must list pytest and xxhash (Goal 07.2)."""
    content: str = (
        Path(__file__).parent.parent / "src" / "requirements.txt"
    ).read_text(encoding="utf-8")
    for dep in ["pytest", "xxhash"]:
        assert dep in content, f"{dep} ausente em requirements.txt"


def test_readme_documents_cache_args() -> None:
    """README.md must document new cache/audit CLI arguments (Goal 07.3)."""
    readme_path: Path = Path(__file__).parent.parent / "README.md"
    content: str = readme_path.read_text(encoding="utf-8")
    required_args: list[str] = [
        "--shared-cache-mb",
        "--cache-key-mode",
        "--cache-key-decimals",
        "--hash-strategy",
        "--audit-json",
        "--audit-tile-csv",
    ]
    missing: list[str] = []
    for arg in required_args:
        if arg not in content:
            missing.append(arg)
    assert not missing, (
        f"README.md is missing documentation for: {', '.join(missing)}"
    )
