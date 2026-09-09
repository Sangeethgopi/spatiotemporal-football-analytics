# REAL DATA INVENTORY

## Summary
Three genuine Metrica Sports professional optical tracking matches are available.
All are at 25 fps, normalised coordinates (0–1), pitch 105×68 m.

| Property | Match 1 | Match 2 | Match 3 |
| :--- | :--- | :--- | :--- |
| **Frames** | 145,006 | 141,156 | 143,761 |
| **Duration** | 96.7 min | 94.1 min | 95.8 min |
| **Frame rate** | 25 fps | 25 fps | 25 fps |
| **Ball coverage** | 60.9% | 59.0% | 67.8% |
| **Turnover events (3m)** | 102 | 78 | 84 |
| **Turnover events (uncoupled)** | 123 | 102 | 109 |
| **Positive frames (target)** | 10,175 | 7,799 | 8,376 |
| **Target prevalence** | 7.0% | 5.5% | 5.8% |
| **Player coordinates** | Yes | Yes | Yes |
| **Ball coordinates** | Yes (60.9%) | Yes (59.0%) | Yes (67.8%) |
| **Team identities** | Anonymised | Anonymised | Anonymised |
| **Ground truth possession** | Derived | Derived | Derived |

## Notes
- Ball coverage of 59–68% is vastly superior to the broadcast CV pipeline's 26.5% (median run 2 frames vs 338 frames here).
- All three matches are independent (different matches, no temporal overlap).
- Match 2 was used as **TRAIN**, Match 3 as **VAL + TEST**.
- Match 1 was used for the synthetic simulation baseline and is preserved as a reference.

## Experimental Design
```
TRAIN : Metrica Match 2 (full, 141,156 frames)
VAL   : Metrica Match 3, frames 1 → 70% split − 100-frame purge (100,532 frames)
TEST  : Metrica Match 3, frames 70% split → end (43,129 frames)  [held out until final]
```

> [!IMPORTANT]
> The TEST set (Match 3 final 30%) was completely untouched during all model selection, smoothing selection, and threshold selection steps.
