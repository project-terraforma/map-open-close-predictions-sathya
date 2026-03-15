"""
Step 10: Model retraining pipeline.

Sub-commands:
    python -m src.step10_retrain auto-label sf 100   # auto-label 100 SF businesses via web+OSM
    python -m src.step10_retrain drift sf             # check feature distribution drift
    python -m src.step10_retrain refresh sf            # re-verify expired labels
    python -m src.step10_retrain retrain               # full retrain with auto-labels
    python -m src.step10_retrain status                # show label counts, model age, drift
"""
