"""Measure end-to-end Lipla recognition throughput."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import lipla


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--provider", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    providers = ["CPUExecutionProvider"] if args.provider == "cpu" else None
    recognizer = lipla.Recognizer(providers=providers)
    workload = args.images * args.iterations

    for _ in range(args.warmup):
        for image in args.images:
            recognizer(image)

    started = time.perf_counter()
    for image in workload:
        recognizer(image)
    elapsed = time.perf_counter() - started

    session_providers = {
        "pose": recognizer.pose_model.session.get_providers(),
        "ocr-det": recognizer.ocr_model.det_session.get_providers(),
        "ocr-rec": recognizer.ocr_model.rec_session.get_providers(),
    }
    print(f"provider mode: {args.provider}")
    print(f"session providers: {session_providers}")
    print(f"images: {len(workload)}")
    print(f"elapsed: {elapsed:.3f} s")
    print(f"throughput: {len(workload) / elapsed:.3f} images/s")


if __name__ == "__main__":
    main()
