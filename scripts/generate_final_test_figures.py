"""Generate final-test figures from the frozen test-result JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from signal_modulation.experiment import prepare_new_run_directory
from signal_modulation.reporting import (
    load_result,
    render_confusion_matrix_svg,
    render_snr_accuracy_svg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_result", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_directory = prepare_new_run_directory(args.output_directory)
    result = load_result(args.test_result)
    snr_path = render_snr_accuracy_svg(
        result,
        output_directory / "final_test_accuracy_by_snr.svg",
        section="test",
        title="TemporalCNN Final Test Accuracy by SNR",
        series_label="Final test",
    )
    confusion_path = render_confusion_matrix_svg(
        result,
        output_directory / "final_test_confusion_matrix.svg",
        section="test",
        title="TemporalCNN Final Test Confusion Matrix",
    )
    print(f"snr_figure={snr_path}")
    print(f"confusion_figure={confusion_path}")


if __name__ == "__main__":
    main()
