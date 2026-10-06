"""Predict the genre of one audio file.  Usage:  python predict_song.py --file path/to/song.wav"""
import argparse
import logging
import sys

from src.config import load_config
from src.utils import set_global_seed, setup_logging

logger = logging.getLogger("predict_song")


def _fmt(conf) -> str:
    return "n/a" if conf is None else f"{conf * 100:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Genre prediction for a single audio file.")
    parser.add_argument("--file", required=True, help="Path to a .wav (or other librosa-readable) file.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_global_seed(cfg.seed, tensorflow=True)

    from src.predict import load_inference_assets, predict_file
    from src.preprocessing import AudioLoadError

    try:
        assets = load_inference_assets(cfg)
        res = predict_file(args.file, assets, cfg)
    except (FileNotFoundError, AudioLoadError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    print("\nCNN Prediction:")
    print(f"Genre: {res['cnn_genre']}")
    print(f"Confidence: {_fmt(res['cnn_confidence'])}")
    print(f"\nHybrid Model Prediction ({res['hybrid_model']}):")
    print(f"Genre: {res['hybrid_genre']}")
    print(f"Confidence: {_fmt(res['hybrid_confidence'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
