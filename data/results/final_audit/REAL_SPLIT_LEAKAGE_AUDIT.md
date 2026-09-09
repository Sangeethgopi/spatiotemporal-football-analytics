# REAL SPLIT LEAKAGE AUDIT

## 1. Split Definition
The dataset consists of Metrica Match 3 (total 143,761 frames).
- `val_split` index = 100,632 (70% of match)
- **Validation**: Frames 0 to 100,531 (100,532 frames)
- **Purge gap**: Frames 100,532 to 100,631 (100 frames)
- **Test**: Frames 100,632 to 143,760 (43,129 frames)

## 2. Overlap Audit
- Mathematical verification confirms `val_frames + purge_frames + test_frames = 143,761`.
- The sets are strictly disjoint.
- **Any overlap between VAL and TEST**: False.

## 3. Boundary Leakage Check
- **GT Turnovers in VAL**: 61
- **GT Turnovers in TEST**: 23
- **TEST turnover labels bleeding into VAL**: 0
- **TEST turnover labels bleeding into Purge**: 0

The 100-frame backward projection (horizon) from turnovers in the TEST set never extends past frame 100,632 into the VAL set, nor into the purge gap.

## 4. Conclusion
**SAFE.** The chronological holdout is clean. The 100-frame purge successfully isolates the validation targets from the test targets. There is zero label leakage across the boundary.
