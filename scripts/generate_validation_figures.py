"""Generate validation figures from two saved experiment-result JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path

from signal_modulation.experiment import prepare_new_run_directory
from signal_modulation.reporting import (
    load_result,
    render_confusion_matrix_svg,
    render_snr_comparison_svg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_result", type=Path)
    parser.add_argument("candidate_result", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_directory = prepare_new_run_directory(args.output_directory)
    baseline = load_result(args.baseline_result)
    candidate = load_result(args.candidate_result)
    snr_path = render_snr_comparison_svg(
        baseline,
        candidate,
        output_directory / "validation_accuracy_by_snr.svg",
    )
    confusion_path = render_confusion_matrix_svg(
        candidate,
        output_directory / "temporal_cnn_confusion_matrix.svg",
    )
    print(f"snr_figure={snr_path}")
    print(f"confusion_figure={confusion_path}")


if __name__ == "__main__":
    main()
