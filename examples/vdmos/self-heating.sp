VDMOS self heating test
M1 D G 0 tj tc IRFP240 thermal
rthk tc 0 0.05
VG G 0 5V Pulse 0 10 0 1m 1m 100m 200m
*RD D D1 4
VD D 0 2V
.model IRFP240 VDMOS nchan
+ Vto=4 Kp=5.9 Lambda=.001 Theta=0.015 ksubthres=.27
+ Rd=61m Rs=18m Rg=3 Rds=1e7
+ Cgdmax=2.45n Cgdmin=10p a=0.3 Cgs=1.2n
+ Is=60p N=1.1 Rb=14m XTI=3
+ Cjo=1.5n Vj=0.8 m=0.5
+ tcvth=0.0065 MU=-1.27 texp0=1.5
+ Rthjc=0.02 Cthj=1e-3 Rthca=1000
+ mtriode=0.8
.control
dc vd 0.1 50 .1 vg 5 13 2
plot -i(vd)
settype temperature v(tj) v(tc)
plot v(tj) v(tc)
*tran 1m 0.01
*plot v(d) v(g)
.endc
.end
