#!/usr/bin/env python3
"""
Comprehensive benchmark script for hash table cache configurations.

Runs the pipeline with different cache configurations on the same input,
measures execution time, cache hit/miss rates, and compares output similarity
against a baseline run.

Usage:
    uv run python3 benchmark_cache.py [--input TIF] [--samples TXT] [--baseline-workers N]
