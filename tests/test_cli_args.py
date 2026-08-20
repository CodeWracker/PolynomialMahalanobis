import subprocess
import sys


def run_main(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pipeline.main", *args],
        capture_output=True, text=True,
    )


def test_round_without_decimals_fails() -> None:
    result = run_main(
        "in.tif", "out.tif", "samples.txt", "log.log",
        "--cache-key-mode", "round",
    )
    assert result.returncode != 0
    assert "cache-key-decimals" in result.stderr


def test_exact_with_decimals_fails() -> None:
    result = run_main(
        "in.tif", "out.tif", "samples.txt", "log.log",
        "--cache-key-mode", "exact", "--cache-key-decimals", "2",
    )
    assert result.returncode != 0
    assert "cache-key-decimals" in result.stderr


def test_decimals_zero_rejected_by_argparse_type() -> None:
    result = run_main(
        "in.tif", "out.tif", "samples.txt", "log.log",
        "--cache-key-mode", "round", "--cache-key-decimals", "0",
    )
    assert result.returncode != 0


def test_negative_cache_mb_fails() -> None:
    result = run_main(
        "in.tif", "out.tif", "samples.txt", "log.log",
        "--shared-cache-mb", "-1",
    )
    assert result.returncode != 0


def test_invalid_hash_strategy_rejected() -> None:
    result = run_main(
        "in.tif", "out.tif", "samples.txt", "log.log",
        "--hash-strategy", "md5",
    )
    assert result.returncode != 0
