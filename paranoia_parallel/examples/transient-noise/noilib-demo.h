
* standard inverter made noisy
*.subckt inv1 dd ss sub well in out
*vn1 out outi dc 0 noise 0.1 0.3n 1.0 0.1
*mn1  outi in  ss  sub  n1  w=2u  l=0.25u  AS=3p AD=3p PS=4u PD=4u
*mp1  outi in  dd  well  p1  w=4u l=0.25u  AS=7p AD=7p PS=6u PD=6u
*.ends inv1

* standard no noise inverter
.subckt inv1 dd ss sub well in out
mn1  out in  ss  sub  n1  w=2u  l=0.25u  AS=3p AD=3p PS=4u PD=4u
mp1  out in  dd  well  p1  w=4u l=0.25u  AS=7p AD=7p PS=6u PD=6u
.ends inv1

* standard no noise inverter
.subckt inv2 dd ss sub well in out
mn1  out in  ss  sub  n1  w=5u  l=0.25u  AS=7p AD=7p PS=7u PD=7u
mp1  out in  dd  well  p1  w=10u l=0.25u  AS=12p AD=12p PS=12u PD=12u
.ends inv2


* very noisy inverter (noise on vdd and well)
.subckt inv1_1 dd ss sub well in out
vn1 dd idd dc 0 trnoise 0.05 0.05n 1 0.05
vn2 well iwell dc 0 trnoise 0.05 0.05n 1 0.05
mn1  out in  ss  sub  n1  w=2u  l=0.25u  AS=3p AD=3p PS=4u PD=4u
mp1  out in  idd  iwell  p1  w=4u l=0.25u  AS=7p AD=7p PS=6u PD=6u
*Cout out 0 0.1p
.ends inv1_1


* another very noisy inverter
.subckt inv1_2 dd ss sub well in out
vn1 out outi dc 0 trnoise 0.05 8p 1.0 0.001
mn1  outi in  ss  sub  n1  w=2u  l=0.25u  AS=3p AD=3p PS=4u PD=4u
mp1  outi in  dd  well  p1  w=4u l=0.25u  AS=7p AD=7p PS=6u PD=6u
*Cout out 0 0.1p
.ends inv1_2

* another very noisy inverter with current souces parallel to transistor
.subckt inv13 dd ss sub well in outi
in1 ss outi dc 0 noise 200u 0.05n 1.0 50u
mn1  outi in  ss  sub  n1  w=2u  l=0.25u  AS=3p AD=3p PS=4u PD=4u
in2 dd outi dc 0 noise 200u 0.05n 1.0 50u
mp1  outi in  dd  well  p1  w=4u l=0.25u  AS=7p AD=7p PS=6u PD=6u
*Cout out 0 0.1p
.ends inv13

.subckt inv53 dd ss sub well in out
xinv1 dd ss sub well in 1 inv1
xinv2 dd ss sub well 1  2 inv1
xinv3 dd ss sub well 2  3 inv1
xinv4 dd ss sub well 3  4 inv1
xinv5 dd ss sub well 4 out inv1
.ends inv53

.subckt inv253 dd ss sub well in out
xinv1 dd ss sub well in 1 inv53
xinv2 dd ss sub well 1  2 inv53
xinv3 dd ss sub well 2  3 inv53
xinv4 dd ss sub well 3  4 inv53
xinv5 dd ss sub well 4 out inv53
.ends inv253
