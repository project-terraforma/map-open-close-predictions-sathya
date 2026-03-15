import sys
from src.step4_classifier import train, predict_db

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"

    if cmd == "predict":
        predict_db()
    elif cmd == "build-deltas":
        from src.step4_classifier.delta_features import build
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
        build(max_per_class_per_city=limit)
    else:
        train()
