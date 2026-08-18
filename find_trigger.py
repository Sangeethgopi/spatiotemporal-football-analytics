import pandas as pd
from src.pipeline import run_pipeline

if __name__ == "__main__":
    print("Running pipeline...")
    res = run_pipeline(save_artifacts=False)
    top = res["top_triggers"]
    print("\n\n---TOP TRIGGER---")
    if not top.empty:
        t = top.iloc[0]
        print(f"Frame: {t['frame']}")
        print(f"Time (s): {t['time_s']:.2f}")
        print(f"Probability: {t['prob']:.3f}")
    else:
        print("No triggers found.")
