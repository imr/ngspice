Test of VDMOS gate-source and gate-drain capacitance

m1 d g s IXTP6N100D2

.MODEL IXTP6N100D2 VDMOS(KP=2.9 RS=0.1 RD=1.3 RG=1 VTO=-2.7 LAMBDA=0.03 CGDMAX=3000p CGDMIN=2p CGS=2915p a=1 TT=1371n IS=2.13E-08 N=1.564 RB=0.0038 m=0.548 Vj=0.1 Cjo=3200pF ksubthres=0.1)

vd d 0 dc 5
vg g 0 pwl (0 -3 1 3)
vs s 0 0

.control
save  all @m1[cgd] @m1[cgs]
tran 1m 1
plot vs#branch
plot @m1[cgd] @m1[cgs]
.endc

.end