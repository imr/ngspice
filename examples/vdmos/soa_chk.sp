VDMOS SOA check

.model IRFP240 VDMOS nchan
+ Vto=4 Kp=5.9 Lambda=.001 Theta=0.015 ksubthres=.27
+ Rd=61m Rs=18m Rg=3 Rds=1e7
+ Cgdmax=2.45n Cgdmin=10p a=0.3 Cgs=1.2n
+ Is=60p N=1.1 Rb=14m Cjo=1.5n XTI=3
+ tcvth=0.0065 MU=-1.27 texp0=1.5
+ mtriode=0.8
+ Vgs_max=20 Vgd_max=20 Vds_max=200

vd1  d1 0 dc 0.1
vg1  g1 0 dc 0.0
vs1  s1 0 dc 0.0
m1  d1 g1 s1 IRFP240

.model IRFP9240 VDMOS pchan
+ Vto=-4 Kp=8.8 Lambda=.003 Theta=0.08 ksubthres=.35
+ Rd=180m Rs=50m Rg=3 Rds=1e7
+ Cgdmax=1.25n Cgdmin=50p a=0.23 Cgs=1.15n
+ Is=150p N=1.3 Rb=16m Cjo=1.3n XTI=2
+ tcvth=0.004 MU=-1.27 texp0=1.5
+ mtriode=0.6
+ Vgs_max=20 Vgd_max=20 Vds_max=200

vd2  0 d2 dc 0.1
vg2  0 g2 dc 0.0
vs2  0 s2 dc 0.0
m2  d2 g2 s2 IRFP9240

.options warn=1 maxwarns=6

.control
dc vd1 -1 210 1 vg1 5 25 5
plot -i(vd1)
dc vd2 -1 210 1 vg2 5 25 5
plot i(vd2)
.endc

.end
