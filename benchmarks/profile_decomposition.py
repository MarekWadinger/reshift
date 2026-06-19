"""Profile time distribution in the OnlineDMD + Rolling pipeline.

Instruments each component to measure where wall-clock time is spent,
so we can estimate realistic Rust speedup potential.
"""

import sys
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path

import numpy as np
from river.decomposition import OnlineDMD
from river.decomposition.osvd import OnlineSVDZhang
from river.decomposition.rust_rolling_dmd import RustRollingDMD

sys.path.append(str(Path(__file__).parent.parent))

from functions.rolling import Rolling

# --- Monkey-patch instrumentation ---

_timings: dict[str, list[float]] = defaultdict(list)


def _instrument(cls, method_name, label):
    original = getattr(cls, method_name)

    @wraps(original)
    def timed(self, *args, **kwargs):
        t0 = time.perf_counter_ns()
        result = original(self, *args, **kwargs)
        _timings[label].append(time.perf_counter_ns() - t0)
        return result

    setattr(cls, method_name, timed)


# Instrument OnlineSVDZhang
_instrument(OnlineSVDZhang, "update", "svd.update")
_instrument(OnlineSVDZhang, "revert", "svd.revert")

# Instrument OnlineDMD
_instrument(OnlineDMD, "_update_A_P", "dmd._update_A_P")
_instrument(OnlineDMD, "_truncate_w_svd", "dmd._truncate_w_svd")
_instrument(OnlineDMD, "update", "dmd.update")
_instrument(OnlineDMD, "revert", "dmd.revert")

# --- Rolling wrapper timing ---

_instrument(Rolling, "update", "rolling.update")


# --- Simulate the pipeline ---
def run_pipeline(m: int, r: int, window_size: int, n_samples: int):
    """Run Hankelizer | Rolling(OnlineDMD) pipeline on synthetic data."""
    rng = np.random.default_rng(42)

    init_size = 300
    dmd = OnlineDMD(
        r=r,
        initialize=init_size,
        w=1.0,
        exponential_weighting=False,
    )
    rolling = Rolling(dmd, window_size=window_size)

    # Generate synthetic data: m-dimensional vectors (as if from Hankelizer)
    data = rng.standard_normal((n_samples, m))

    # Measure dict creation overhead separately
    t_dict_create = 0
    t_total_start = time.perf_counter_ns()

    for i in range(n_samples):
        t0 = time.perf_counter_ns()
        x = {f"x{j}": data[i, j] for j in range(m)}
        t_dict_create += time.perf_counter_ns() - t0
        rolling.update(x=x)

    t_total = time.perf_counter_ns() - t_total_start
    _timings["dict_creation"] = [t_dict_create]

    # Also try numpy-direct path for comparison (separate timing)
    saved_timings = {k: list(v) for k, v in _timings.items()}
    _timings.clear()
    dmd2 = OnlineDMD(
        r=r,
        initialize=init_size,
        w=1.0,
        exponential_weighting=False,
    )
    rolling2 = Rolling(dmd2, window_size=window_size)
    t_np_start = time.perf_counter_ns()
    for i in range(n_samples):
        rolling2.update(x=data[i : i + 1])
    t_np_total = time.perf_counter_ns() - t_np_start
    _timings.clear()
    _timings.update(saved_timings)
    _timings["numpy_direct_total"] = [t_np_total]

    return t_total


def run_rust_pipeline(m: int, r: int, window_size: int, n_samples: int):
    """Run RustRollingDMD pipeline on synthetic data."""
    rng = np.random.default_rng(42)

    init_size = 300
    data = rng.standard_normal((n_samples, m))

    # Warmup: one throwaway instance to avoid first-call overhead
    warmup = RustRollingDMD(
        r=r,
        w=1.0,
        window_size=window_size,
        initialize=init_size,
        exponential_weighting=False,
    )
    for i in range(min(10, n_samples)):
        warmup.update(x=data[i : i + 1])
    del warmup

    rust_dmd = RustRollingDMD(
        r=r,
        w=1.0,
        window_size=window_size,
        initialize=init_size,
        exponential_weighting=False,
    )

    t_total_start = time.perf_counter_ns()
    for i in range(n_samples):
        rust_dmd.update(x=data[i : i + 1])
    t_total = time.perf_counter_ns() - t_total_start

    return t_total


def print_results(t_total, n_samples):
    print(f"\n{'=' * 70}")
    print(f"Pipeline: {n_samples} samples, total = {t_total / 1e6:.1f} ms")
    print(f"Per sample: {t_total / n_samples / 1e3:.1f} µs")
    print(f"{'=' * 70}")
    print(
        f"{'Component':<25} {'Total ms':>10} {'Per call µs':>12} {'Calls':>8} {'% Total':>8}",
    )
    print(f"{'-' * 25} {'-' * 10} {'-' * 12} {'-' * 8} {'-' * 8}")

    for label in [
        "rolling.update",
        "dmd.update",
        "dmd.revert",
        "dmd._truncate_w_svd",
        "dmd._update_A_P",
        "svd.update",
        "svd.revert",
    ]:
        times = _timings.get(label, [])
        if not times:
            continue
        total_ns = sum(times)
        count = len(times)
        per_call_us = total_ns / count / 1e3
        total_ms = total_ns / 1e6
        pct = total_ns / t_total * 100
        print(
            f"{label:<25} {total_ms:>10.1f} {per_call_us:>12.1f} {count:>8} {pct:>7.1f}%",
        )

    # Compute "other" time (framework overhead not in any instrumented method)
    rolling_total = sum(_timings.get("rolling.update", []))
    dmd_update_total = sum(_timings.get("dmd.update", []))
    dmd_revert_total = sum(_timings.get("dmd.revert", []))
    dmd_inner = dmd_update_total + dmd_revert_total
    rolling_overhead = rolling_total - dmd_inner
    print(
        f"\n{'Rolling overhead':<25} {rolling_overhead / 1e6:>10.1f} {'':>12} {'':>8} {rolling_overhead / t_total * 100:>7.1f}%",
    )

    # Dict conversion overhead estimate
    # dmd.update includes dict conversion + _truncate_w_svd + _update_A_P
    truncate_total = sum(_timings.get("dmd._truncate_w_svd", []))
    update_ap_total = sum(_timings.get("dmd._update_A_P", []))
    svd_update_total = sum(_timings.get("svd.update", []))
    svd_revert_total = sum(_timings.get("svd.revert", []))

    # In dmd.update: dict conversion + init checks + _truncate_w_svd + _update_A_P
    dmd_update_overhead = dmd_update_total - truncate_total - update_ap_total
    # Subtract the part of _update_A_P that's called from revert too
    # Actually _update_A_P is called once in update and once in revert
    # _truncate_w_svd includes svd.update
    truncate_overhead = truncate_total - svd_update_total

    print(
        f"{'dmd.update overhead':<25} {dmd_update_overhead / 1e6:>10.1f} {'':>12} {'':>8} {dmd_update_overhead / t_total * 100:>7.1f}%",
    )
    print("  (dict conv, init, vstack)")
    print(
        f"{'_truncate overhead':<25} {truncate_overhead / 1e6:>10.1f} {'':>12} {'':>8} {truncate_overhead / t_total * 100:>7.1f}%",
    )
    print("  (matrix rotations, np.linalg.inv)")

    # Estimate Rust-portable fraction
    rust_portable = (
        svd_update_total
        + svd_revert_total
        + update_ap_total
        + truncate_overhead
    )
    print(f"\n{'=' * 70}")
    print(
        f"Rust-portable compute:    {rust_portable / 1e6:>10.1f} ms = {rust_portable / t_total * 100:.1f}% of total",
    )
    print(
        f"Python framework:         {(t_total - rust_portable) / 1e6:>10.1f} ms = {(t_total - rust_portable) / t_total * 100:.1f}% of total",
    )
    print("\nIf Rust kernels are 100x faster:")
    rust_fast = rust_portable / 100
    new_total = (t_total - rust_portable) + rust_fast
    speedup = t_total / new_total
    print(
        f"  New total: {new_total / 1e6:.1f} ms ({t_total / 1e6:.1f} -> {new_total / 1e6:.1f})",
    )
    print(f"  End-to-end speedup: {speedup:.1f}x")

    dict_create = sum(_timings.get("dict_creation", [0]))
    dict_create = sum(_timings.get("dict_creation", [0]))
    print(
        f"{'Dict creation (caller)':<25} {dict_create / 1e6:>10.1f} {'':>12} {'':>8} {dict_create / t_total * 100:>7.1f}%",
    )

    np_total = sum(_timings.get("numpy_direct_total", [0]))
    print(
        f"\nNumpy-direct path (no dict): {np_total / 1e6:.1f} ms = {t_total / np_total:.1f}x faster than dict path",
    )

    # Neatly aligned output for breakdown table
    print(f"\n{'=' * 70}")
    print("DETAILED BREAKDOWN")
    print(f"{'=' * 70}")

    # Helper for better table-style alignment (labels left, ms right, pct right)
    def fmt(label, ms, percent):
        return f"{label:<45} {ms:>12.1f} ms   {percent:>7.2f}%"

    def line(label, val):
        return fmt(label, val / 1e6, val / t_total * 100)

    print(line("svd.update  (QR, SVD, matmul on ~80x2):", svd_update_total))
    print(line("svd.revert  (SVD, matmul on ~3x3):", svd_revert_total))
    print(line("_truncate matrix ops (inv, @):", truncate_overhead))
    print(line("_update_A_P (inv, @, sym on 2x2):", update_ap_total))
    print("-" * 70)
    print(
        line(
            "Numerical compute total:",
            svd_update_total
            + svd_revert_total
            + truncate_overhead
            + update_ap_total,
        ),
    )
    print(
        line("dmd.update overhead (dict, vstack, init):", dmd_update_overhead),
    )
    print(line("Rolling overhead (deque, revert dispatch):", rolling_overhead))
    print(line("Dict creation in caller:", dict_create))
    unaccounted = t_total - (
        svd_update_total
        + svd_revert_total
        + truncate_overhead
        + update_ap_total
        + dmd_update_overhead
        + rolling_overhead
        + dict_create
    )
    # dmd.revert overhead (dict conv + _truncate + _update_A_P already counted from update side)
    # dmd_revert_overhead = (
    #     dmd_revert_total
    #     - (truncate_total - svd_update_total)
    #     - (update_ap_total / 2)
    # )
    # Align overhead line likewise
    print(
        f"{'  dmd.revert overhead (dict, _x_first):':<45} {unaccounted / 1e6:>12.1f} ms   {(unaccounted / t_total * 100):>7.2f}%",
    )

    print("\nIf ALSO move dict conv + Rolling to Rust (fused pipeline):")
    rust_portable_full = rust_portable + dmd_update_overhead + rolling_overhead
    rust_fast_full = rust_portable_full / 100
    new_total_full = (t_total - rust_portable_full) + rust_fast_full
    speedup_full = t_total / new_total_full
    print(
        f"  Rust-portable: {rust_portable_full / 1e6:.1f} ms = {rust_portable_full / t_total * 100:.1f}%",
    )
    print(f"  New total: {new_total_full / 1e6:.1f} ms")
    print(f"  End-to-end speedup: {speedup_full:.1f}x")


if __name__ == "__main__":
    # Match typical usage: m=80, r=2, window=301
    M = 80
    R = 2
    WINDOW = 302
    N = 10000  # enough samples for stable measurement

    print(f"Profiling: m={M}, r={R}, window={WINDOW}, n_samples={N}")
    t_total = run_pipeline(M, R, WINDOW, N)
    print_results(t_total, N)

    # --- Rust benchmark ---
    print(f"\n{'=' * 70}")
    print("RUST BENCHMARK")
    print(f"{'=' * 70}")
    t_rust = run_rust_pipeline(M, R, WINDOW, N)

    py_ms = t_total / 1e6
    py_us = t_total / N / 1e3
    rs_ms = t_rust / 1e6
    rs_us = t_rust / N / 1e3
    speedup = t_total / t_rust if t_rust > 0 else float("inf")

    print(f"Python: {py_ms:.1f} ms ({py_us:.1f} \u00b5s/sample)")
    print(f"Rust:   {rs_ms:.1f} ms ({rs_us:.1f} \u00b5s/sample)")
    print(f"Speedup: {speedup:.1f}x")
