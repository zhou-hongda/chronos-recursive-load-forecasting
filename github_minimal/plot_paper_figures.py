"""Generate manuscript figures from the existing Proj6 experimental exports."""

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def positive_dpi(value):
    dpi = int(value)
    if dpi <= 0:
        raise argparse.ArgumentTypeError("DPI must be positive")
    return dpi


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=int, choices=[2, 3, 4], nargs="+",
                        default=[2, 3, 4], help="Figure numbers to regenerate (default: 2, 3, 4; Figure 1 is supplied separately)")
    parser.add_argument("--dpi", type=positive_dpi, default=600,
                        help="PNG resolution; physical figure size is unchanged (default: 600)")
    parser.add_argument("--formats", choices=["png", "pdf", "svg"], nargs="+",
                        default=["png", "pdf", "svg"], help="Publication export formats")
    parser.add_argument("--mse-display", choices=["relative", "thousands", "raw"],
                        default="relative", help="Existing relative MSE, MSE/1000, or original MSE")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/paper/manuscript_figures",
                        help="Output directory; relative paths are resolved from the project root")
    return parser


def main():
    args = build_parser().parse_args()
    from paper_figures.manuscript import main as generate

    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    generate(figures=args.figures, output_dir=output, dpi=args.dpi,
             formats=args.formats, mse_display=args.mse_display)


if __name__ == "__main__":
    main()
