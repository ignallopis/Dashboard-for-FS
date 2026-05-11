# CAT17x — Vehicle Parameters (Parameters.m)

## Vehicle geometry
| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Wheelbase | `Vhcl.Wheel_Base` | 1.53 | m |
| Front track | `Vhcl.tf` | 1.225 | m |
| Rear track | `Vhcl.tr` | 1.175 | m |
| Wheel radius | `Vhcl.Wheel_Radius` | 0.2032 | m |
| Gear ratio | `Vhcl.i` | 9.05 | — |
| CoG height | `Vhcl.CoG_z` | 0.278 | m |
| CoG lateral | `Vhcl.CoG_y` | 0 | m |
| Rear weight dist. | `Vhcl.distrib_rear` | 0.50 | — (50/50) |
| lf (CoG→front axle) | `Vhcl.lf` | 0.765 | m |
| lr (CoG→rear axle) | `Vhcl.lr` | 0.765 | m |

## Mass
| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Car mass | `Vhcl.m_vhcl` | 220 | kg |
| Pilot mass | `Vhcl.m_pilot` | 68 | kg |
| Total mass | `Vhcl.m` | 288 | kg |
| Front axle mass | `Vhcl.mf` | 144 | kg |
| Rear axle mass | `Vhcl.mr` | 144 | kg |
| Yaw inertia Iz | `Vhcl.Iz` | 129.024 | kg·m² |
| Wheel inertia | `Vhcl.Wheel_Inertia` | 0.145429 | kg·m² |

## Suspension
| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Front roll centre height | `Vhcl.hrcf` | 0.012 | m |
| Rear roll centre height | `Vhcl.hrcr` | 0.042 | m |
| Front roll stiffness | `Vhcl.Krollf` | 36 929.4 | N·m/rad |
| Rear roll stiffness | `Vhcl.Krollr` | 40 833.7 | N·m/rad |

Roll stiffness ratio front/total = 36929 / (36929 + 40834) = **47.5 % front** → slight rear-biased roll stiffness.

## Aerodynamics
| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Reference area | `Vhcl.A` | 1.0 | m² |
| Drag coefficient | `Vhcl.Coef_Drag` | 1.803 | — |
| Lift coefficient | `Vhcl.Coef_Lift` | −5.913 | — (downforce) |
| CoP x (from front axle) | `Vhcl.CoP_x` | −0.7547 | m |
| CoP z (from front axle) | `Vhcl.CoP_z` | 0.557 | m |
Legacy
## Safety limits
| Parameter | Value | Units |
|-----------|-------|-------|
| Max motor torque | 27.5 | N·m |
| Max regen torque | −27.5 | N·m |
| Max RPM (motors) | 16 000 | rpm |
| Max RPM (FW) | 9 000 | rpm |
| Motor OT error | 120 | °C |
| Inverter OT error | 75 | °C |

## Power Control
| Parameter | Value | Units |
|-----------|-------|-------|
| Max power | 80 000 | W |
| Max battery regen current | 80 | A |

## Brake
| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Front piston count | `brake.nf` | 8 | — |
| Rear piston count | `brake.nr` | 4 | — |
| Piston diameter | `brake.dp` | 0.023 | m |
| Pad outer radius | `brake.Re` | 0.0927 | m |
| Pad inner radius | `brake.Ri` | 0.0608 | m |
| Friction coefficient | `brake.mud` | 0.617 | — |
| Front balance | `brake.frontBalance` | 0.67 | — |
| Rear balance | `brake.rearBalance` | 0.33 | — |
| Max deceleration | `brake.MaxDeceleration` | 1.79 | g |

TODO: add front and rear heave stiffness (`k_heave_F`, `k_heave_R`) to enable a theoretical Aero Load · Heave vs Speed overlay in Dynamics.

## Sensor positions (from front axle, positive forward/left/up)
| Sensor | dx [m] | dz [m] |
|--------|--------|--------|
| SBGF (front IMU) | +0.252 | +0.579 |
| SBGR (rear IMU) | −1.603 | +0.350 |
| VN (VectorNav) | −0.392 | +0.072 |
