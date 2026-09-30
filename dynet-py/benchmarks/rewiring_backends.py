"""Compare dense/sparse rewiring on reproducible synthetic edge lists.

Run from dynet-py:
    PYTHONPATH=src python benchmarks/rewiring_backends.py --nodes 500 --networks 8

Input generation and preparation are excluded. First-call time includes backend
cache construction; cached time is the median of repeated calculations. Peak
traced allocation is measured separately on a fresh prepared collection and is
not total process RSS. No benchmark timing is used as a test assertion.
"""

import argparse
import gc
import json
import statistics
import time
import tracemalloc

import numpy as np
import pandas as pd

from dynet_py import prepare_networks, rewiring_analysis


def make_inputs(nodes, networks, density):
    rng = np.random.default_rng(20260930)
    labels = np.array([f"node_{i}" for i in range(nodes)])
    result = {}
    for index in range(networks):
        positions = rng.choice(nodes**2, size=round(nodes**2 * density), replace=False)
        result[str(index)] = pd.DataFrame({
            # Zero-weight loops retain all nodes, including isolates.
            "from": np.concatenate((labels[positions // nodes], labels)),
            "to": np.concatenate((labels[positions % nodes], labels)),
            "weight": np.concatenate((rng.uniform(0.1, 3., len(positions)), np.zeros(nodes))),
        })
    return result


def timed(prepared, backend):
    start = time.perf_counter()
    result = rewiring_analysis(prepared, backend=backend)
    return (time.perf_counter() - start) * 1000, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=300)
    parser.add_argument("--networks", type=int, default=6)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.001, 0.01, 0.1, 0.5])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.nodes < 1 or args.networks < 2 or args.repeats < 1 or any(not 0 <= d <= 1 for d in args.densities):
        parser.error("Require nodes >= 1, networks >= 2, repeats >= 1, and densities in [0, 1].")
    for density in args.densities:
        inputs = make_inputs(args.nodes, args.networks, density)
        reference = None
        for backend in ("dense", "sparse"):
            prepared = prepare_networks(inputs)
            first_ms, result = timed(prepared, backend)
            if reference is None:
                reference = result
            else:
                pd.testing.assert_frame_equal(result, reference, rtol=1e-10, atol=1e-10)
            samples = [timed(prepared, backend)[0] for _ in range(args.repeats)]
            del prepared
            gc.collect()
            prepared = prepare_networks(inputs)
            tracemalloc.start()
            rewiring_analysis(prepared, backend=backend)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            print(json.dumps({
                "backend": backend, "networks": args.networks, "nodes": args.nodes,
                "density": density, "first_ms": round(first_ms, 3),
                "cached_median_ms": round(statistics.median(samples), 3),
                "peak_traced_mib": round(peak / 1024**2, 3), "repeats": args.repeats,
                "auto_backend": prepared._rewiring_backend(),
            }), flush=True)
            del prepared


if __name__ == "__main__":
    main()
