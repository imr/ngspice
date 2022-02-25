Capacitance and current comparison between models d and bulk diode in vdmos

D1 ad kd dio
.model dio d TT=1371n IS=2.13E-08 N=1.564 RS=0.0038 m=0.548 Vj=0.1 Cjo=3200pF

Va ad 0 ac 1 dc 0.5 pwl(0 -2 2.5 0.5)
Vk kd 0 0

m1 d g s IXTP6N100D2
.MODEL IXTP6N100D2 VDMOS(KP=2.9 RS=0.1 RD=1.3 RG=1 VTO=-2.7 LAMBDA=0.03 CGDMAX=3000p CGDMIN=2p CGS=2915p a=1 TT=1371n IS=2.13E-08 N=1.564 RB=0.0038 m=0.548 Vj=0.1 Cjo=3200pF ksubthres=0.1)

Vd d 0 ac 1 dc -0.5 pwl(0 2 2.5 -0.5)
Vg g 0 -5  ; transistor is off
Vs s 0 0

.control
save all @d1[id] @m1[id] @d1[cd] @m1[cds] all
tran 10m 2.5
plot abs(i(Vk)) abs(i(Vs)) ylog
plot @d1[cd] @m1[cds]
*plot abs(i(Vk)) - abs(i(Vs))
*plot @d1[cd] - @m1[cds]

ac dec 10 1 100K
plot mag(i(Vs)) mag (i(Vk))
plot ph(i(Vs)) ph(i(Vk))
.endc

.end
