# Fx Pacejka — Longitudinal Force (12-coef, Camber & Pressure)

## Inputs

| Variable | Description | Units | Valor para calculos |
|----------|-------------|-------|
| `SL` | Slip Ratio (longitudinal) — positive = driving force | dimensionless [0,1] | obtain from csv [Est_SRFL,Est_SRFR,Est_SRRL,Est_SRRR] |
| `SA` | Slip Angle — positive = right driving force | rad | obtain from csv [Est_SAFL,Est_SAFR,Est_SARL,Est_SARR] |
| `FZ` | Vertical load | N | obtain from csv [Est_FZFL,Est_FZFR,Est_FZRL,Est_FZRR] |
| `IA` | Camber (inclination) angle | rad | 0.013962634 |
| `P` | Tyre pressure | kPa | 70 |

## Reference values

| Parameter | Value |
|-----------|-------|
| `F_Z0` — nominal vertical load | 714.329 N |
| `P_0` — nominal pressure | 77.723 kPa |

## Coefficients

| Name | Value |
|------|-------|
| `Cx` | 1.334221905727055 |
| `pDx1` | 2.599999999998234 |
| `pDx2` | −0.083334621896074 |
| `pDx3` | 10.012674685003649 |
| `Bx` | 12.309225807476469 |
| `pEx1` | −0.910253269775273 |
| `pEx2` | 1.061553952705465 |
| `Dyx` | 1.000000000000000 |
| `Cyx` | 0.849999929264725 |
| `Byx` | 10.276843593534068 |
| `Sx` | 0 |
| `pPx1` | −0.372205525624779 |

## Formulation

```
dfz = (FZ − F_Z0) / F_Z0
dp  = (P  − P_0 ) / P_0

Dx  = (pDx1 + pDx2·dfz) · (1 − pDx3·IA²) · (1 + pPx1·dp)
Ex  = pEx1 + pEx2·dfz

FXp = FZ · Dx · sin( Cx · atan( Bx·SL − Ex·(Bx·SL − atan(Bx·SL)) ) )

Gx  = Dyx · cos( Cyx · atan( Byx · atan(SA) ) )

FX  = Gx · FXp + Sx
```

`FXp` is the pure-slip longitudinal force. `Gx` is the combined-slip reduction factor that attenuates `FX` when a slip angle is present.
