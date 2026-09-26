"""Command line entry point: ``carprice audit`` and ``carprice train``."""

from __future__ import annotations

import argparse
import json

from threadpoolctl import threadpool_limits

from .data import ROOT
from .models import N_JOBS


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="carprice")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit", help="audit the original 2022 notebook")
    t = sub.add_parser("train", help="run the full experiment and export web assets")
    t.add_argument("--n-boot", type=int, default=2000)
    t.add_argument("--quick", action="store_true", help="skip hyperparameter search")
    args = p.parse_args(argv)

    reports = ROOT / "reports"
    with threadpool_limits(limits=N_JOBS):
        if args.cmd == "audit":
            from .audit import run

            rep = run(reports)
            print(json.dumps({k: v for k, v in rep.items() if k != "rerun_with_fixes"}, indent=2))
            rr = rep["rerun_with_fixes"]
            print(f"re-run R2 over seeds: mean {rr['r2_mean']:.3f} [{rr['r2_min']:.3f}, {rr['r2_max']:.3f}]")
        else:
            from .train import run

            rep = run(reports, ROOT / "web", n_boot=args.n_boot, quick=args.quick)
            print((reports / "results.md").read_text())
            print(json.dumps(rep["dataset"], indent=2))
            print(json.dumps(rep["export_parity"], indent=2))
            print("runtime", rep["runtime_seconds"], "s")


if __name__ == "__main__":
    main()
