import sys
from pathlib import Path
from src.step4_classifier import train, predict_db, retrain_pipeline, PROJECT_ROOT

def eval_model(city: str = None):
    """Evaluate model against ground truth labels — no web crawler, no API calls.

    Uses the same fixed ground truth businesses each time so you can track
    model improvement across retrains.
    """
    import pickle
    import json as _json
    import pandas as pd
    from src.step4_classifier import extract_features, MODEL_PATH
    from src.config import engine
    from sqlalchemy import text

    if not MODEL_PATH.exists():
        print("No model found. Run: python -m src.step4_classifier")
        return

    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    model, feature_cols, threshold = saved["model"], saved["feature_cols"], saved["threshold"]

    # Get ground truth businesses
    query = """
        SELECT g.overture_id, o.name, g.is_open, o.confidence, o.raw_json
        FROM ground_truth.labels g
        JOIN overture.places o ON o.id = g.overture_id
    """
    params = {}
    if city:
        aliases = {"sf": "san_francisco", "nyc": "new_york", "chi": "chicago"}
        key = aliases.get(city, city)
        query += " WHERE g.city = :city"
        params["city"] = key

    query += " ORDER BY o.name"

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()

    if not rows:
        print(f"No ground truth found{f' for {city}' if city else ''}. Run step7 first.")
        return

    # Build features for each business
    records = []
    labels = []
    names = []
    for ov_id, name, is_open, confidence, raw_json in rows:
        record = {"confidence": confidence, "base_confidence": confidence}
        rj = raw_json
        if rj:
            if isinstance(rj, str):
                try:
                    rj = _json.loads(rj)
                except (ValueError, TypeError):
                    rj = {}
            if isinstance(rj, dict):
                for key in ("sources", "names", "categories", "websites", "socials",
                             "emails", "phones", "brand", "addresses"):
                    val = rj.get(key)
                    if val is not None:
                        serialized = _json.dumps(val) if not isinstance(val, str) else val
                        record[key] = serialized
                        record[f"base_{key}"] = serialized
        records.append(record)
        labels.append(1 if is_open else 0)
        names.append(name)

    df = pd.DataFrame(records)
    feat = extract_features(df)
    for col in feature_cols:
        if col not in feat.columns:
            feat[col] = 0

    X = feat[feature_cols].values.astype(float)
    probas = model.predict_proba(X)[:, 1]
    preds = (probas >= threshold).astype(int)
    labels_arr = pd.array(labels)

    # Results table
    correct = 0
    total = len(rows)
    print(f"\n{'='*70}")
    print(f"MODEL EVALUATION — {total} ground truth businesses")
    print(f"{'='*70}")
    print(f"{'#':<4} {'Business':<35} {'Actual':<8} {'Pred':<8} {'Score':<7} {'Result'}")
    print("-" * 70)

    for i, (name, actual, pred, prob) in enumerate(zip(names, labels, preds, probas), 1):
        actual_str = "OPEN" if actual == 1 else "CLOSED"
        pred_str = "OPEN" if pred == 1 else "CLOSED"
        match = "OK" if pred == actual else "WRONG"
        if pred == actual:
            correct += 1
        print(f"{i:<4} {name[:35]:<35} {actual_str:<8} {pred_str:<8} {prob:<7.3f} {match}")

    acc = 100 * correct / total
    print(f"\n{'='*70}")
    print(f"Model accuracy: {correct}/{total} = {acc:.1f}%")
    print(f"Threshold: {threshold:.3f}")

    # Per-class breakdown
    tp = sum(1 for a, p in zip(labels, preds) if a == 1 and p == 1)
    fn = sum(1 for a, p in zip(labels, preds) if a == 1 and p == 0)
    fp = sum(1 for a, p in zip(labels, preds) if a == 0 and p == 1)
    tn = sum(1 for a, p in zip(labels, preds) if a == 0 and p == 0)
    print(f"  Open:   {tp}/{tp+fn} correct ({100*tp/(tp+fn):.0f}% recall)" if tp+fn > 0 else "")
    print(f"  Closed: {tn}/{tn+fp} correct ({100*tn/(tn+fp):.0f}% recall)" if tn+fp > 0 else "")
    print(f"{'='*70}")

    # ── Append to eval history log ──
    from datetime import datetime
    history_path = PROJECT_ROOT / "eval_history.csv"

    # Count feedback labels
    fb_path = PROJECT_ROOT / "feedback_labels.parquet"
    if fb_path.exists():
        fb = pd.read_parquet(fb_path)
        n_feedback = len(fb)
    else:
        n_feedback = 0

    # Write header if new file
    if not history_path.exists():
        with open(history_path, "w") as f:
            f.write("timestamp,city,accuracy,correct,total,open_recall,closed_recall,threshold,feedback_labels\n")

    open_recall = round(100 * tp / (tp + fn)) if (tp + fn) > 0 else 0
    closed_recall = round(100 * tn / (tn + fp)) if (tn + fp) > 0 else 0

    with open(history_path, "a") as f:
        f.write(f"{datetime.now().isoformat()},{city or 'all'},{acc:.1f},{correct},{total},"
                f"{open_recall},{closed_recall},{threshold:.3f},{n_feedback}\n")

    print(f"\nResults saved to eval_history.csv ({history_path.name})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"

    if cmd == "predict":
        predict_db()
    elif cmd == "eval":
        city = sys.argv[2] if len(sys.argv) > 2 else None
        if city == "all":
            # Eval each city separately so you can track per-city improvement
            from sqlalchemy import text as sa_text
            from src.config import engine
            with engine.connect() as conn:
                cities = conn.execute(sa_text(
                    "SELECT DISTINCT city FROM ground_truth.labels"
                )).fetchall()
            city_keys = [r[0] for r in cities]
            for c in sorted(city_keys):
                eval_model(c)
            print(f"\nEvaluated {len(city_keys)} cities. See eval_history.csv for trends.")
        else:
            eval_model(city)
    elif cmd == "foursquare":
        from src.step4_classifier.foursquare_labels import generate_labels
        city = sys.argv[2] if len(sys.argv) > 2 else "sf"
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 500
        if city == "all":
            for c in ["san_francisco", "new_york", "chicago"]:
                generate_labels(c, limit)
        else:
            aliases = {"sf": "san_francisco", "nyc": "new_york", "chi": "chicago"}
            generate_labels(aliases.get(city, city), limit)
    elif cmd == "yelp":
        from src.step4_classifier.yelp_labels import generate_labels
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
        generate_labels(limit)
    elif cmd == "retrain":
        retrain_pipeline()
    elif cmd == "build-deltas":
        from src.step4_classifier.delta_features import build
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
        build(max_per_class_per_city=limit)
    elif cmd == "clear-feedback":
        fb_path = PROJECT_ROOT / "feedback_labels.parquet"
        if fb_path.exists():
            import pandas as pd
            fb = pd.read_parquet(fb_path)
            n_open = (fb["label"] == 1).sum()
            n_closed = (fb["label"] == 0).sum()
            fb_path.unlink()
            print(f"Deleted feedback_labels.parquet ({len(fb)} labels: {n_open} open, {n_closed} closed)")
        else:
            print("No feedback_labels.parquet found — nothing to clear")
    elif cmd == "show-feedback":
        fb_path = PROJECT_ROOT / "feedback_labels.parquet"
        if fb_path.exists():
            import pandas as pd
            fb = pd.read_parquet(fb_path)
            n_open = (fb["label"] == 1).sum()
            n_closed = (fb["label"] == 0).sum()
            print(f"Feedback labels: {len(fb)} total ({n_open} open, {n_closed} closed)")
            if "city" in fb.columns:
                for city in fb["city"].unique():
                    c = fb[fb["city"] == city]
                    print(f"  {city}: {len(c)} ({(c['label']==1).sum()} open, {(c['label']==0).sum()} closed)")
        else:
            print("No feedback_labels.parquet found")
    else:
        train()
