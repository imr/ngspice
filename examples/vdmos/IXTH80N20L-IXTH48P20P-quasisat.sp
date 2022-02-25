VDMOS Test of quasi saturation IXTH80N20L IXTH48P20P
* Original VDMOS model parameters taken from David Zan,
* http://www.diyaudio.com/forums/software-tools/266655-power-mosfet-models-ltspice-post5300643.html
* The Quasi-saturation is added for demonstration only, it is not aligned with the data sheets
* and is for sure exaggerated, at least for the IXTH80N20L

mn1 d1 g1 s1 IXTH80N20L

vd1 d1 0 1
vg1 g1 0 1
vs1 s1 0 0

mp2 d2 g2 s2 IXTH48P20P

vd2 d2 0 1
vg2 g2 0 1
vs2 s2 0 0

.control
dc vd1 -1 100 0.05 vg1 3 10 1
altermod mn1 rq=0
altermod mn1 Lambda=2m
dc vd1 -1 100 0.05 vg1 3 10 1
plot dc1.vs1#branch vs1#branch

dc vd2 1 -100 -0.05 vg2 -3 -10 -1
altermod mp2 rq=0
altermod mp2 Lambda=5m
dc vd2 1 -100 -0.05 vg2 -3 -10 -1
plot dc3.vs2#branch vs2#branch

.endc

* David Zan, (c) 2017/03/02 Preliminary
.MODEL IXTH80N20L VDMOS Nchan Vds=200
+ VTO=4 KP=15
+ Lambda=3m  ; will be reset by altermod to original 2m
+ Mtriode=0.4
+ Ksubthres=120m
+ subshift=160m
+ Rs=5m Rd=10m Rds=200e6
+ Cgdmax=9000p Cgdmin=300p A=0.25
+ Cgs=5500p Cjo=11000p
+ Is=10e-6 Rb=8m   
+ BV=200 IBV=250e-6
+ NBV=4
+ TT=250e-9
+ vq=100
+ rq=0.5  ; will be reset by altermod to original 0

* David Zan, (c) 2017/03/02 Preliminary
.MODEL IXTH48P20P VDMOS Pchan Vds=200
+ VTO=-4 KP=10
+ Lambda=7m  ; will be reset by altermod to original 5m  
+ Mtriode=0.3   
+ Ksubthres=120m
+ Rs=10m Rd=20m Rds=200e6
+ Cgdmax=6000p Cgdmin=100p A=0.25
+ Cgs=5000p Cjo=9000p
+ Is=2e-6 Rb=20m   
+ BV=200 IBV=250e-6
+ NBV=4
+ TT=260e-9
+ vq=100
+ rq=0.5  ; will be reset by altermod to original 0

.end