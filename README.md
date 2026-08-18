# ⚽ Pressing Intensity & Trigger Detection Thesis Pipeline

An end-to-end sports analytics research pipeline and interactive web application for **"Tracking-Derived Time-Resolved Pressing Intensity and Press Trigger Detection"**, benchmarked against the traditional **PPDA (Passes Allowed Per Defensive Action)** baseline using **Metrica Sports Open Tracking Data (25 Hz)**.

---

## 🚀 Quickstart

### 1. Run the Full End-to-End Pipeline
Execute data loading, turnover derivation, spatial-temporal feature engineering, PPDA calculation, ML model training, and baseline evaluation:

```bash
python src/pipeline.py
```

### 2. Launch the Interactive Streamlit Web Dashboard
Launch the interactive 2D pitch tracking explorer, pressing timeline, and thesis analytics app:

```bash
streamlit run app.py
```

### 3. Run Automated Tests
Run unit and integration test suites:

```bash
python -m unittest discover tests
```

---

## 📁 Repository Structure

```
pressing_intensity_thesis/
├── data/                       # Match tracking CSVs and exported results
├── src/
│   ├── __init__.py
│   ├── config.py               # Pitch dimensions (105x68m), 25Hz sampling, windows
│   ├── download_metrica.py     # Metrica open data fetcher & synthetic generator
│   ├── data_loader.py          # Metrica 3-row header parser, metric normalizer
│   ├── possession.py           # Nearest player possession & turnover derivation
│   ├── features.py             # Convex hull, closing velocity, line height, pressure index
│   ├── baseline_ppda.py        # Team PPDA & rolling proximity baseline classifier
│   ├── model.py                # Gradient Boosting & Random Forest classifiers
│   ├── evaluate.py             # Precision, Recall, F1, ROC-AUC, PR curves
│   └── pipeline.py             # Orchestrated pipeline runner
├── app.py                      # Interactive Streamlit Thesis Dashboard
├── thesis/
│   ├── research_plan.md        # Hypotheses, variables, research questions
│   ├── methodology.md          # Full mathematical derivations and formulas
│   └── thesis_draft.md         # Thesis paper writeup draft
├── tests/
│   └── test_pipeline.py        # Automated test suite (5/5 passing)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🔬 Core Features & Methodology

1. **Self-Contained Tracking Pipeline:** Assigns ball possession per frame and derives forced turnovers without needing fragile external event-data synchronization.
2. **Kinematic Feature Suite:**
   - **Defender Closing Speed:** Rate of approach vector ($-\frac{dd}{dt} = \mathbf{v}_d \cdot \hat{\mathbf{u}}$).
   - **Team Compactness:** 2D Convex Hull area and spatial dispersion ($\sigma_x, \sigma_y$).
   - **Defensive Line Height:** Depth compression of defensive line.
   - **Pressure Index:** Composite physical formula for multi-defender press urgency.
3. **PPDA Baseline Benchmark:** Traditional PPDA calculation and proximity thresholding benchmark.
4. **Machine Learning Model:** Supervised Gradient Boosted Decision Trees with balanced class weights on out-of-sample temporal validation splits.
5. **Interactive Dashboard:** 2D animated pitch view, 15-minute intensity profiles, press trigger moment browser, and LaTeX export generator.
