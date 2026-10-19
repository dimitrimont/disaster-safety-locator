"""Train the app's models on every landfall and save them to Hurricane_model/.

    python scripts/train.py              # forest and network (needs TensorFlow)
    python scripts/train.py --no-network
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hurricane.data import load_landfalls  # noqa: E402
from hurricane.model import FEATURES, MODEL_DIR, TARGET, save, train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--no-network", action="store_true")
args = ap.parse_args()
df = load_landfalls()
print("forest ->", save(train(df)))
if not args.no_network:
    from hurricane.deep import SizeNetwork

    SizeNetwork(seed=0).fit(df[FEATURES].astype(float), df[TARGET].astype(float)).save(MODEL_DIR)
    print("network ->", MODEL_DIR)
