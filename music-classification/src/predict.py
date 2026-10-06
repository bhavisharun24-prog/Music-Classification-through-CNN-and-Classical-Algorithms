"""Inference: identical preprocessing to training + CNN and best hybrid prediction."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .cnn_feature_extractor import build_feature_extractors
from .cnn_model import check_cnn_input, load_best_cnn
from .config import Config
from .preprocessing import compute_mfcc, load_audio_fixed, prepare_cnn_input
from .utils import load_joblib, load_json

logger = logging.getLogger(__name__)


def best_model_file(cfg: Config) -> Path:
    return cfg.classical_dir / "best_model.json"


def load_inference_assets(cfg: Config) -> dict:
    """Load CNN, feature extractors, normalisation stats and the selected hybrid classifier."""
    info_path = cfg.features_dir / "dataset_info.json"
    stats_path = cfg.features_dir / "mfcc_norm_stats.npz"
    if not info_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError("Feature artifacts missing. Run: python -m src.feature_extraction")
    info = load_json(info_path)
    if info["signature"] != cfg.extraction_signature():
        raise ValueError("config.yaml audio/genre settings differ from those used in training.")
    if not best_model_file(cfg).is_file():
        raise FileNotFoundError("No hybrid model selected. Run: python train_hybrid.py")
    best = load_json(best_model_file(cfg))
    stats = np.load(stats_path)
    cnn = load_best_cnn(cfg)
    return {
        "class_names": info["class_names"], "mean": stats["mean"], "std": stats["std"],
        "cnn": cnn, "extractors": build_feature_extractors(cnn), "best": best,
        "hybrid": load_joblib(cfg.artifacts_dir / best["model_path"]),
    }


def _top(probs: np.ndarray, classes: np.ndarray, names: list[str]) -> tuple[str, float]:
    k = int(np.argmax(probs))
    return names[int(classes[k])], float(probs[k])


def predict_file(path: Path | str, assets: dict, cfg: Config) -> dict:
    """Predict the genre of one audio file with the CNN and with the best hybrid model."""
    y = load_audio_fixed(path, cfg)                       # same loader as training
    x = prepare_cnn_input(compute_mfcc(y, cfg), assets["mean"], assets["std"])  # train-only stats
    check_cnn_input(x, cfg)

    names = assets["class_names"]
    cnn_probs = assets["cnn"].predict(x, verbose=0)[0]
    cnn_genre, cnn_conf = _top(cnn_probs, np.arange(len(names)), names)

    source = assets["best"]["feature_key"]
    feats = assets["extractors"][source].predict(x, verbose=0)
    pipe = assets["hybrid"]
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(feats)[0]
        hyb_genre, hyb_conf = _top(probs, pipe.classes_, names)
    else:
        hyb_genre, hyb_conf = names[int(pipe.predict(feats)[0])], None
    return {"cnn_genre": cnn_genre, "cnn_confidence": cnn_conf,
            "hybrid_genre": hyb_genre, "hybrid_confidence": hyb_conf,
            "hybrid_model": f"{assets['best']['model_label']} on {assets['best']['feature_source']}"}
