#!/usr/bin/env python3
"""Comprehensive benchmark for PythonPolinomial."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np

_SYS_PATH_ADDED = False
NL = chr(10)
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC = _SCRIPT_DIR / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
    _SYS_PATH_ADDED = True


@dataclass
class BenchResult:
    config_desc: str
    shared_cache_mb: float
    hash_strategy: str
    cache_key_mode: str
    cache_key_decimals: int
    workers: int
    wall_clock_s: float
    hit_rate: float
    miss_empty: int
    miss_collision: int
    inserted: int
    similarity: float
    diff_pct: float
    max_pixel_diff: int
    mean_diff: float


def _ensure_src_path(base_dir: Path) -> None:
    src = base_dir / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
        global _SYS_PATH_ADDED
        _SYS_PATH_ADDED = True


def _build_matrix() -> list[dict[str, Any]]:
    cache_sizes = [0, 16, 64, 128, 256, 512]
    strategies = ["blake2b", "xxhash"]
    workers_list = [1, 2, 4]
    round_decimals = [1, 2, 3, 4, 5]
    matrix: list[dict[str, Any]] = []
    for cache_mb in cache_sizes:
        for workers in workers_list:
            if cache_mb == 0:
                # baseline/no-cache: only exact
                matrix.append({
                    "shared_cache_mb": 0,
                    "hash_strategy": "",
                    "workers": workers,
                    "cache_key_mode": "exact",
                    "cache_key_decimals": None,
                })
            else:
                for strat in strategies:
                    # exact mode
                    matrix.append({
                        "shared_cache_mb": cache_mb,
                        "hash_strategy": strat,
                        "workers": workers,
                        "cache_key_mode": "exact",
                        "cache_key_decimals": None,
                    })
                    # round mode with each decimal
                    for dec in round_decimals:
                        matrix.append({
                            "shared_cache_mb": cache_mb,
                            "hash_strategy": strat,
                            "workers": workers,
                            "cache_key_mode": "round",
                            "cache_key_decimals": dec,
                        })
    return matrix


def _config_desc(cfg: dict[str, Any]) -> str:
    workers = cfg["workers"]
    cache = cfg["shared_cache_mb"]
    strat = cfg.get("hash_strategy", "")
    mode = cfg.get("cache_key_mode", "exact")
    dec = cfg.get("cache_key_decimals", None)
    mode_part = mode
    if dec is not None:
        mode_part = f"{mode}d{dec}"
    if cache == 0:
        return f"no-cache_w{workers}_{mode_part}"
    return f"cache{cache}MB_{strat}_w{workers}_{mode_part}"
def _run_pipeline(
    input_path: Path,
    samples_path: Path,
    output_dir: Path,
    cfg: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    run_dir = output_dir / _config_desc(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_tif = run_dir / "output.tif"
    audit_json = run_dir / "audit.json"
    cmd: list[str] = [
        sys.executable, "src/main.py",
        str(input_path), str(output_tif), str(samples_path), str(run_dir / "benchmark.log"),
        "--order", "3",
        "--exp", "-1.0",
        "--tile-size", "1024",
        "--workers", str(cfg["workers"]),
        "--shared-cache-mb", str(cfg["shared_cache_mb"]),
        "--cache-key-mode", cfg["cache_key_mode"],
    ]
    dec = cfg.get("cache_key_decimals")
    if dec is not None:
        cmd.append("--cache-key-decimals")
        cmd.append(str(dec))
    if cfg["shared_cache_mb"] > 0 and cfg["hash_strategy"]:
        cmd.append("--hash-strategy")
        cmd.append(cfg["hash_strategy"])
    cmd.append("--audit-json")
    cmd.append(str(audit_json))
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "42"
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        err_file = run_dir / "pipeline_error.log"
        err_file.write_text(f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}")
        raise RuntimeError(f"Pipeline failed for {_config_desc(cfg)} with rc={proc.returncode}. See {err_file}")
    audit: dict[str, Any] = {}
    if audit_json.exists():
        with open(audit_json) as f:
            audit = json.load(f)
    return wall, audit


def _compare_with_baseline(
    baseline_tif: Path,
    result_tif: Path,
) -> dict[str, float]:
    from compare.outputs import compare_outputs
    cmp = compare_outputs(str(baseline_tif), str(result_tif), tile_size=1024)
    return {
        "similarity": cmp.global_similarity,
        "diff_pct": (cmp.total_diff_pixels / cmp.total_pixels * 100) if cmp.total_pixels else 0.0,
        "max_pixel_diff": cmp.global_max_diff,
        "mean_diff": cmp.global_mean_diff,
    }
def _generate_plots(results: list[BenchResult], outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    cache_sizes = sorted(set(r.shared_cache_mb for r in results))
    worker_groups = sorted(set((r.workers, r.hash_strategy) for r in results))
    for i, (workers, strat) in enumerate(worker_groups):
        subset = [r for r in results if r.workers == workers and r.hash_strategy == strat]
        if not subset:
            subset = [r for r in results if r.workers == workers and r.shared_cache_mb == 0]
        if not subset:
            continue
        xs = [r.shared_cache_mb for r in subset]
        ys = [r.wall_clock_s for r in subset]
        label = f"w={workers}"
        if strat:
            label += f" ({strat[:4]})"
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("Cache Size (MB)")
    ax.set_ylabel("Wall Clock (s)")
    ax.set_title("Execution Time vs Cache Size by Workers")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "time_vs_cache.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    cached = [r for r in results if r.shared_cache_mb > 0]
    if cached:
        for strat in sorted(set(r.hash_strategy for r in cached)):
            subset = [r for r in cached if r.hash_strategy == strat]
            xs = [r.shared_cache_mb for r in subset]
            ys = [r.hit_rate for r in subset]
            ax.scatter(xs, ys, label=strat, alpha=0.7)
            ax.plot(sorted(xs), [y for _, y in sorted(zip(xs, ys))], alpha=0.5)
    ax.set_xlabel("Cache Size (MB)")
    ax.set_ylabel("Hit Rate")
    ax.set_title("Hit Rate vs Cache Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(plots_dir / "hitrate_vs_cache.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    worker_times = []
    for w in sorted(set(r.workers for r in results)):
        subset = [r for r in results if r.workers == w]
        avg_time = sum(r.wall_clock_s for r in subset) / len(subset)
        worker_times.append(avg_time)
    workers_sorted = sorted(set(r.workers for r in results))
    bar_colors = ["steelblue", "coral", "seagreen"][:len(workers_sorted)]
    bars = ax.bar([str(w) for w in workers_sorted], worker_times, color=bar_colors)
    ax.set_xlabel("Workers")
    ax.set_ylabel("Avg Wall Clock (s)")
    ax.set_title("Workers vs Execution Time")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "workers_vs_time.png", dpi=150)
    plt.close(fig)
def _write_csv(results: list[BenchResult], outdir: Path) -> None:
    fieldnames = [
        "config_desc", "shared_cache_mb", "hash_strategy", "cache_key_mode",
        "cache_key_decimals", "workers",
        "wall_clock_s", "hit_rate", "miss_empty", "miss_collision", "inserted",
        "similarity", "diff_pct", "max_pixel_diff", "mean_diff",
    ]
    csv_path = outdir / "benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "config_desc": r.config_desc,
                "shared_cache_mb": r.shared_cache_mb,
                "hash_strategy": r.hash_strategy,
                "cache_key_mode": r.cache_key_mode or "",
                "cache_key_decimals": r.cache_key_decimals or "",
                "workers": r.workers,
                "wall_clock_s": round(r.wall_clock_s, 4),
                "hit_rate": round(r.hit_rate, 6),
                "miss_empty": r.miss_empty,
                "miss_collision": r.miss_collision,
                "inserted": r.inserted,
                "similarity": round(r.similarity, 8),
                "diff_pct": round(r.diff_pct, 6),
                "max_pixel_diff": r.max_pixel_diff,
                "mean_diff": round(r.mean_diff, 6),
            })


def _format_terminal_table(results: list[BenchResult]) -> str:
    if not results:
        return "No results."
    headers = ["Config", "Cache MB", "Hash", "Mode", "W", "Wall(s)", "HitRate", "Sim", "Diff%"]
    rows: list[list[str]] = []
    for r in results:
        rows.append([
            r.config_desc,
            str(r.shared_cache_mb),
            r.hash_strategy if r.hash_strategy else "-",
            r.cache_key_mode or "-",
            str(r.workers),
            f"{r.wall_clock_s:.2f}",
            f"{r.hit_rate:.4f}",
            f"{r.similarity:.6f}",
            f"{r.diff_pct:.4f}",
        ])
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    def fmt_row(cells: list[str]) -> str:
        return " | ".join(cells[j].ljust(col_widths[j]) for j in range(len(cells)))
    sep = "-+-".join("-" * w for w in col_widths)
    output_lines = [fmt_row(headers), sep]
    for row in rows:
        output_lines.append(fmt_row(row))
    return NL.join(output_lines)
def _generate_markdown_report(results: list[BenchResult], outdir: Path) -> None:
    path = outdir / "RELATORIO_BENCHMARK.md"
    baseline = next((r for r in results if r.shared_cache_mb == 0 and r.workers == 1), None)
    baseline_time = baseline.wall_clock_s if baseline else 1.0
    fastest = min(results, key=lambda r: r.wall_clock_s)
    md: list[str] = [
        "# Relatorio Benchmark",
        "",
        f"**Total configuracoes testadas:** {len(results)}",
        "",
        "## Melhores Configuracoes",
        "",
        "| Rank | Config | Cache MB | Hash | Workers | Temps | Speedup | Sim |",
        "|------|--------|----------|------|---------|-------|---------|-----|",
    ]
    sorted_results = sorted(results, key=lambda r: r.wall_clock_s)
    for i, r in enumerate(sorted_results[:10], 1):
        speedup = baseline_time / r.wall_clock_s if r.wall_clock_s > 0 else 0.0
        md.append(
            f"| {i} | {r.config_desc} | {r.shared_cache_mb} | "
            f"{r.hash_strategy or chr(45)} | {r.workers} | {r.wall_clock_s:.2f} | "
            f"{speedup:.2f}x | {r.similarity:.6f} |"
        )
    md.append("")
    md.append("## Analise de Velocidade")
    md.append("")
    if baseline:
        md.append(f"**Baseline (no-cache, 1 worker):** {baseline.wall_clock_s:.2f}s")
    md.append(f"**Mais rapida:** {fastest.config_desc} — {fastest.wall_clock_s:.2f}s")
    spd = baseline_time / fastest.wall_clock_s if fastest.wall_clock_s > 0 else 0.0
    md.append(f"**Speedup maximo:** {spd:.2f}x")
    md.append("")
    md.append("## Integridade")
    md.append("")
    md.append("| Config | Similaridade | Diff% | MaxDiff |")
    md.append("|--------|-------------|-------|---------|")
    for r in sorted_results[:5]:
        md.append(f"| {r.config_desc} | {r.similarity:.8f} | {r.diff_pct:.6f}% | {r.max_pixel_diff} |")
    md.append("")
    md.append("## Cache Effectiveness")
    md.append("")
    cached = [r for r in results if r.shared_cache_mb > 0]
    if cached:
        md.append("| Config | HitRate | MissEmpty | MissColl | Inserted |")
        md.append("|--------|---------|-----------|----------|----------|")
        for r in sorted(cached, key=lambda x: x.hit_rate, reverse=True):
            md.append(f"| {r.config_desc} | {r.hit_rate:.6f} | {r.miss_empty} | {r.miss_collision} | {r.inserted} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write(NL.join(md))
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark full")
    parser.add_argument("--input", default="tests/fixtures/orthophoto-ms.tif")
    parser.add_argument("--samples", default="tests/fixtures/orthophoto-ms.txt")
    parser.add_argument("--outdir", default="./benchmark_results")
    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / args.input
    samples_path = base_dir / args.samples
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    _ensure_src_path(base_dir)
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if not samples_path.exists():
        print(f"ERROR: samples not found: {samples_path}", file=sys.stderr)
        sys.exit(1)
    matrix = _build_matrix()
    print(f"Total configurations: {len(matrix)}")
    baseline_cfg = {"shared_cache_mb": 0, "hash_strategy": "", "workers": 1, "cache_key_mode": "exact", "cache_key_decimals": None}
    print()
    print(f"[1/{len(matrix) + 1}] BASELINE (no-cache, w=1)...")
    baseline_dir = outdir / _config_desc(baseline_cfg)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_wall, baseline_audit = _run_pipeline(input_path, samples_path, outdir, baseline_cfg)
    baseline_tif = baseline_dir / "output.tif"
    print(f"  Wall: {baseline_wall:.2f}s")
    all_results: list[BenchResult] = []
    for idx, cfg in enumerate(matrix, 2):
        desc = _config_desc(cfg)
        run_dir = outdir / desc
        result_tif_checkpoint = run_dir / "output.tif"
        if result_tif_checkpoint.exists():
            print(f"[{idx}/{len(matrix) + 1}] {desc}... SKIPPED (checkpoint)")
            continue
        print()
        print(f"[{idx}/{len(matrix) + 1}] {desc}...")
        run_dir.mkdir(parents=True, exist_ok=True)
        wall, audit = _run_pipeline(input_path, samples_path, outdir, cfg)
        result_tif = run_dir / "output.tif"
        similarity_data = _compare_with_baseline(baseline_tif, result_tif)
        miss_empty = audit.get("cache_misses_empty", 0)
        miss_collision = audit.get("cache_misses_collision", 0)
        inserted = audit.get("cache_inserted", 0)
        hit_rate = audit.get("hit_rate", 0.0)
        result = BenchResult(
            config_desc=desc,
            shared_cache_mb=cfg["shared_cache_mb"],
            hash_strategy=cfg.get("hash_strategy", ""),
            cache_key_mode=cfg.get("cache_key_mode", "exact"),
            cache_key_decimals=cfg.get("cache_key_decimals", 0) or 0,
            workers=cfg["workers"],
            wall_clock_s=wall,
            hit_rate=hit_rate,
            miss_empty=int(miss_empty),
            miss_collision=int(miss_collision),
            inserted=int(inserted),
            similarity=similarity_data["similarity"],
            diff_pct=similarity_data["diff_pct"],
            max_pixel_diff=similarity_data["max_pixel_diff"],
            mean_diff=similarity_data["mean_diff"],
        )
        all_results.append(result)
        print(f"  Wall: {wall:.2f}s  HitRate: {hit_rate:.4f}  Sim: {result.similarity:.6f}")
    baseline_result = BenchResult(
        config_desc=_config_desc(baseline_cfg),
        shared_cache_mb=0,
        hash_strategy="",
        cache_key_mode="exact",
        cache_key_decimals=0,
        workers=1,
        wall_clock_s=baseline_wall,
        hit_rate=baseline_audit.get("hit_rate", 0.0),
        miss_empty=int(baseline_audit.get("cache_misses_empty", 0)),
        miss_collision=int(baseline_audit.get("cache_misses_collision", 0)),
        inserted=int(baseline_audit.get("cache_inserted", 0)),
        similarity=1.0,
        diff_pct=0.0,
        max_pixel_diff=0,
        mean_diff=0.0,
    )
    all_results.insert(0, baseline_result)
    print()
    print("=" * 80)
    print("Generating outputs...")
    _write_csv(all_results, outdir)
    _generate_plots(all_results, outdir)
    _generate_markdown_report(all_results, outdir)
    print()
    print(_format_terminal_table(all_results))
    print()
    print(f"Artifacts saved to: {outdir}/")


if __name__ == "__main__":
    main()
