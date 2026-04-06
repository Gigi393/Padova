Summary of optimization and scenario validation

Best parameters
- PID (rank 1): kp=0.0002, ki=0.0, kd=0.0, target_bg=110.0
- Basal-bolus (BB, rank 1): cr_multiplier=0.8

Baseline optimization (best settings)
- PID: TIR 95.02%, Hypo 1.03%, CV 15.91%, CompositeScore 95.11
- BB: TIR 93.09%, Hypo 1.20%, CV 12.94%, CompositeScore 94.20
- Takeaway: PID achieves higher TIR and CompositeScore with slightly lower Hypo; BB has lower CV (smoother).

Scenario comparisons (using best settings)

1) Carb overestimate (+20%)
- PID: TIR 95.23%, Hypo 1.34%, CV 16.06%, CompositeScore 95.13
- BB: TIR 93.65%, Hypo 2.02%, CV 14.19%, CompositeScore 94.16
- Edge: PID (better TIR, Hypo, Composite); BB smoother (lower CV).

2) Missed meal input
- PID: TIR 81.31%, Hypo 0.55%, CV 26.31%, CompositeScore 85.99
- BB: TIR 73.34%, Hypo 0.00%, CV 29.21%, CompositeScore 81.08
- Edge: PID (higher TIR, lower CV, higher Composite); BB has zero Hypo but at the cost of lower TIR.

3) Slower sensor
- PID: TIR 93.67%, Hypo 1.84%, CV 17.52%, CompositeScore 93.90
- BB: TIR 93.09%, Hypo 1.20%, CV 12.94%, CompositeScore 94.20
- Edge: BB (lower Hypo, much lower CV, slightly higher Composite); PID has slightly higher TIR.

Tradeoffs and recommendation
- PID consistently delivers higher TIR and CompositeScore in baseline and 2/3 scenarios, with slightly higher variability (CV) and modestly higher Hypo in the slower-sensor case.
- BB tends to be smoother (lower CV) and sometimes slightly safer on Hypo, but generally with lower TIR and CompositeScore.
- Overall stronger: PID, because it wins on CompositeScore in baseline and in 2 of 3 scenarios, and maintains excellent TIR with low Hypo.

recommended_params
{
  "pid": {
    "kp": 0.0002,
    "ki": 0.0,
    "kd": 0.0,
    "target_bg": 110.0
  },
  "bb": {
    "cr_multiplier": 0.8
  }
}