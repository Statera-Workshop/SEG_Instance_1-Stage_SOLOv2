"""Train seg SOLOv2 + BCE mask loss, then evaluate on the test split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main
from detectors.solov2_bce.build import build_model

if __name__ == "__main__":
    main("train", task="seg", version="solov2_bce", build_model=build_model)
