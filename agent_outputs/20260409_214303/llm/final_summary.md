Optimization summary
- Best PID (rank 1): kp=0.00018, ki=0.0, kd=2e-05, target_bg=115
  - TIR 61.48%, Hypo 5.41%, CV 43.85%, CompositeScore 70.88
- Best BB (rank 1): cr_multiplier=1.085
  - TIR 83.21%, Hypo 2.42%, CV 23.23%, CompositeScore 86.88

Baseline optimization comparison
- BB clearly outperforms PID: much higher TIR (83.21 vs 61.48), lower Hypo (2.42 vs 5.41), much lower CV (23.23 vs 43.85), and higher CompositeScore (86.88 vs 70.88).

Scenario validation (using best parameters above)
- Baseline:
  - BB: TIR 83.21, Hypo 2.42, CV 23.23, Composite 86.88
  - PID: TIR 61.48, Hypo 5.41, CV 43.85, Composite 70.88
  - Result: BB better on all four metrics.
- Carb overestimate +20%:
  - BB: TIR 84.58, Hypo 5.11, CV 26.31, Composite 86.58
  - PID: TIR 61.48, Hypo 5.41, CV 43.85, Composite 70.88
  - Result: BB much higher TIR and lower CV; similar Hypo; stronger overall.
- Carb underestimate −20%:
  - BB: TIR 76.56, Hypo 0.92, CV 21.81, Composite 83.48
  - PID: TIR 61.48, Hypo 5.41, CV 43.85, Composite 70.88
  - Result: BB markedly better across all four metrics.
- Missed meal input:
  - BB: TIR 39.12, Hypo 0.0, CV 44.01, Composite 59.07
  - PID: TIR 61.48, Hypo 5.41, CV 43.85, Composite 70.88
  - Result: PID maintains control without meal input (higher TIR, Composite); BB avoids hypoglycemia but loses TIR and Composite.
- Slower sensor:
  - BB: same as baseline (TIR 83.21, Hypo 2.42, CV 23.23, Composite 86.88)
  - PID: TIR 58.49, Hypo 7.82, CV 48.62, Composite 67.94
  - Result: BB robust; PID degrades (lower TIR, higher Hypo and CV).

Key tradeoffs
- BB delivers consistently higher TIR, lower Hypo, and much lower variability (CV) in all scenarios except missed meal input, where reliance on meal announcements causes a large TIR drop despite zero Hypo.
- PID is meal-agnostic and steadier when meals are not announced, but it has lower TIR, higher Hypo, and higher CV overall, and it is more sensitive to sensor lag.

Overall recommendation
- Stronger overall: Basal-bolus (BB). It achieves superior TIR, lower Hypo, much lower CV, and higher CompositeScore in 4 of 5 validated scenarios and in baseline optimization. PID is preferred only if missed meal inputs are frequent and unmitigated.

recommended_params
{"pid":{"kp":0.00018,"ki":0.0,"kd":2e-05,"target_bg":115.0},"bb":{"cr_multiplier":1.085}}