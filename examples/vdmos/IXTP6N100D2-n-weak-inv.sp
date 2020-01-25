VDMOS output

m1 d g s IXTP6N100D2
m2 d g s2 IXTP6N100D2_2

* LTSPICE model parameters
*.MODEL IXTP6N100D2 VDMOS(KP=2.9 RS=0.1 RD=1.3 RG=1 VTO=-2.7 LAMBDA=0.03 CGDMAX=3000p CGDMIN=2p CGS=2915p TT=1371n a=1 IS=2.13E-08 N=1.564 RB=0.0038 m=0.548 *Vj=0.1 Cjo=3200pF ksubthres=0.1)

* equivalent ngspice model parameters
.MODEL IXTP6N100D2_2 VDMOS(KP=2.9 RS=0.1 RD=1.3 RG=1 VTO=-2.7 LAMBDA=0.03 CGDMAX=3000p CGDMIN=2p CGS=2915p TT=1371n a=1 IS=2.13E-08 N=1.564 RB=0.0038 m=0.548 Vj=0.1 Cjo=3200pF ksubthres=39m)

* equivalent ngspice model parameters, trying to make output similar to data sheet Fig. 2
.MODEL IXTP6N100D2 VDMOS(KP=6 RS=0.1 RD=1.3 RG=1 VTO=-2.7 LAMBDA=0.007 CGDMAX=3000p CGDMIN=2p CGS=2915p TT=1371n a=1 IS=2.13E-08 N=1.564 RB=0.0038 m=0.548 Vj=0.1 Cjo=3200pF ksubthres=39m rq=4 vq=200 mtriode=0.1)

vd d 0 -0.6
vg g 0 -2.3
vs s 0 0
vs2 s2 0 0

.control
dc  vg -3.1 -2.1 0.01 vd 0.2 1 0.2
plot vs#branch
plot vs#branch ylog
dc  vd 0 60 0.1 vg -3 5 1
plot vs#branch vs2#branch xlimit 0 60 ylimit 0 14
.endc

.end
