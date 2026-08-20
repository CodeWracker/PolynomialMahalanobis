import sys
import pytest
from pathlib import Path

TESTS_DIR: Path = Path(__file__).parent
PROJECT_DIR: Path = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
SRC_DIR: Path = PROJECT_DIR / "src"
GOLDEN_DIR: Path = TESTS_DIR / "golden_master"
FIXTURES_DIR: Path = TESTS_DIR / "fixtures"
TEST_CASES_DIR: Path = PROJECT_DIR / "test_cases"


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    return GOLDEN_DIR


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def golden_12003_tif() -> Path:
    p = GOLDEN_DIR / "12003.tif"
    if not p.exists():
        pytest.skip("Execute goal 00.3 primeiro para capturar golden masters")
    return p


@pytest.fixture(scope="session")
def golden_12003_poly() -> Path:
    p = GOLDEN_DIR / "12003-poly.tif"
    if not p.exists():
        pytest.skip("Execute goal 00.3 primeiro para capturar golden masters")
    return p


@pytest.fixture(scope="session")
def golden_12003_txt() -> Path:
    p = GOLDEN_DIR / "12003.txt"
    if not p.exists():
        pytest.skip("Execute goal 00.3 primeiro para capturar golden masters")
    return p
