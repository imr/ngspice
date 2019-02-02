* VDMOS self heating test
M1 D G 0 t IRFP240
VG G 0 5V Pulse 0 10 0 1m 1m 100m 200m
*RD D D1 4
VD D 0 2V
.model IRFP240 VDMOS
+ Kp=3.5 Vto=4 Lambda=.003 Theta=0.01
+ Rd=52m Rs=18m Rb=36m Rg=3 
+ Cgdmax=1.34n Cgdmin=.1n Cgs=1.25n 
+ Cjo=1.25n Is=67p
+ ksubthres=.1 
+ shmod=1 RTH0=.01 CTH0=1e-5 MU=1.27 texp0=1.5 texp1=0.3 
+ vq=100 rq=500m
.control
*op
*print -i(vd)
*dc vg 0 9 0.1
dc vd 0.1 50 .1 vg 5 13 2
plot -i(vd) xlog ylog ylimit .1 100
settype temperature v(t)
plot v(t)
*tran 1m 0.01
*plot v(d1)
.endc
.end
