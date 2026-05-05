# Fy Pacejka — Lateral Force (12-coef, Camber & Pressure)

## Inputs

| Variable | Description | Units |
|----------|-------------|-------|
| `SL` | Slip Ratio (longitudinal) — positive = driving force | dimensionless [0,1] | obtain from csv [Est_SRFL,Est_SRFR,Est_SRRL,Est_SRRR] |
| `SA` | Slip Angle — positive = right driving force | rad | obtain from csv [Est_SAFL,Est_SAFR,Est_SARL,Est_SARR] |
| `FZ` | Vertical load (positive downward) | N | Obtain from csv [Est_FZFL,Est_FZFR,Est_FZRL,Est_FZRR] |
| `IA` | Camber (inclination) angle | rad | 0.013962634 | 
| `P` | Tyre pressure | kPa | 70 | 

## Reference values

| Parameter | Value |
|-----------|-------|
| `F_Z0` — nominal vertical load | 686.384 N |
| `P_0` — nominal pressure | 77.598 kPa |

## Coefficients

| Name | Value |
|------|-------|
| `Cy` | 1.416628146849960 |
| `pDy1` | 2.922062365723097 |
| `pDy2` | −0.458333591731965 |
| `By0` | 8.897610109557609 |
| `pEy1` | 0.254453572544703 |
| `pEy2` | 0 |
| `Dxy` | 0.898127775222102 |
| `Cxy` | 0.793203031130685 |
| `Bxy` | 21.718392825857475 |
| `pHy` | 0.121333900331434 |
| `pPy1` | −0.200931771090979 |
| `pPy2` | 0.129559286221261 |

## Formulation

```
dfz   = (FZ − F_Z0) / F_Z0
dp    = (P  − P_0 ) / P_0

Dy    = (pDy1 + pDy2·dfz) · (1 + pPy1·dp)
Ey    = pEy1 + pEy2·dfz
By    = By0 · (1 + pPy2·dp)

SA_eq = SA + pHy·IA          ← camber-equivalent slip angle shift

FYp   = FZ · Dy · sin( Cy · atan( By·SA_eq − Ey·(By·SA_eq − atan(By·SA_eq)) ) )

Gy    = Dxy · cos( Cxy · atan( Bxy · SL ) )

FY    = −(Gy · FYp)
```

`FYp` is the pure-slip lateral force. The camber angle `IA` enters via the horizontal shift `pHy·IA` in `SA_eq`. `Gy` is the combined-slip reduction factor that attenuates `FY` when a slip ratio is present. The sign convention is negated so that a positive slip angle produces a force in the expected direction.
