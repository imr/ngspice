crss coss ciss
*
VP1 P1 0 PULSE(0 1.15m 100n 10n 10n 1 2)
VP2 P4 0 PULSE(0 2.8m 100n 10n 10n 1 2)
*
M1 d1 g1 0 IRFP240
V1 g1 0 0.0
V2 1 d1 0.0
G1 0 1 P1 0 1.04
*
M2 d2 0 d2 IRFP240
V3 2 d2 0.0
G2 0 2 P4 0 1.1
*
M3 d3 g3 0 IRFP9240
V4 g3 0 0.0
V5 3 d3 0.0
G3 3 0 P1 0 0.85
e1 d1p 0 d3 0 -1
*
M4 d4 0 d4 IRFP9240
V6 4 d4 0.0
G4 4 0 P4 0 1.0
e2 d2p 0 d4 0 -1
*
.control
tran 1n 25u
*plot v(d1) v(d2) v(d3) v(d4)

plot 'i(v1)/deriv(v(d1))' 'i(v2)/deriv(v(d1))' vs v(d1) xlog xlimit 1 100 ylimit 0 3n title "IRFP240 crss & coss"
plot 'i(v3)/deriv(v(d2))' vs v(d2) xlog xlimit 1 100 ylimit 0 3n title "IRFP240 ciss"

plot 'i(v4)/deriv(v(d3))' 'i(v5)/deriv(v(d3))' vs v(d1p) xlog xlimit 1 100 ylimit 0 3n title "IRFP9240 crss & coss"
plot 'i(v6)/deriv(v(d4))' vs v(d2p) xlog xlimit 1 100 ylimit 0 3n title "IRFP9240 ciss"

.endc
.model IRFP240 VDMOS nchan
+ Vto=4 Kp=5.9 Lambda=.001 Theta=0.015 ksubthres=.27
+ Rd=61m Rs=18m Rg=3 Rds=1e7
+ Cgdmax=2.45n Cgdmin=10p a=0.3 Cgs=1.2n
+ Is=60p N=1.1 Rb=14m XTI=3
+ Cjo=1.5n Vj=0.8 m=0.5
+ tcvth=0.0065 MU=-1.27 texp0=1.5
*+ Rthjc=0.4 Cthj=5e-3
+ mtriode=0.8
.model IRFP9240 VDMOS pchan
+ Vto=-4 Kp=8.8 Lambda=.003 Theta=0.08 ksubthres=.35
+ Rd=180m Rs=50m Rg=3 Rds=1e7
+ Cgdmax=1.25n Cgdmin=50p a=0.23 Cgs=1.15n
+ Is=150p N=1.3 Rb=16m XTI=2
+ Cjo=1.3n Vj=0.8 m=0.5
+ tcvth=0.004 MU=-1.27 texp0=1.5
*+ Rthjc=0.4 Cthj=5e-3
+ mtriode=0.6
.end

