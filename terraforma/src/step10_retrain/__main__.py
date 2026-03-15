import sys
from src.step10_retrain.auto_label import auto_label
from src.step10_retrain.drift import check_drift
from src.step10_retrain.retrain import retrain

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.step10_retrain diff <city>              # Overture monthly diff labels (fast, free)")
        print("  python -m src.step10_retrain auto-label <city> [count]  # Web+OSM auto-labels (slower, accurate)")
        print("  python -m src.step10_retrain drift <city>               # Check feature distribution drift")
        print("  python -m src.step10_retrain refresh <city> [batch]     # Re-verify expired labels")
        print("  python -m src.step10_retrain retrain                    # Retrain model with all labels")
        print("  python -m src.step10_retrain status                     # Show label counts and model info")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "diff":
        city = sys.argv[2] if len(sys.argv) > 2 else "sf"
        from src.step10_retrain.overture_diff import store_diff_labels
        # Map short names to db names
        city_map = {"sf": "san_francisco", "nyc": "new_york", "chicago": "chicago"}
        db_city = city_map.get(city, city)
        store_diff_labels(db_city)

    elif cmd == "auto-label":
        city = sys.argv[2] if len(sys.argv) > 2 else "sf"
        count = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        auto_label(city, count)

    elif cmd == "drift":
        city = sys.argv[2] if len(sys.argv) > 2 else "sf"
        check_drift(city)

    elif cmd == "refresh":
        city = sys.argv[2] if len(sys.argv) > 2 else "sf"
        batch = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        from src.step10_retrain.auto_label import refresh_stale
        refresh_stale(city, batch)

    elif cmd == "retrain":
        retrain()

    elif cmd == "status":
        from src.step10_retrain.status import show_status
        show_status()

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

main()
