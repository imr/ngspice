Simple regulated DCDC step-up converter

V1 clock  0      PULSE(0 6 0 19u 1u 10n 20.01u)
V2 ref    0      2.5
R1 OUT    outdiv 100K
R2 0      outdiv 27k
R3 outdiv x      10k
C2 err    x      50n
B1 err    0      V = max(0,min(5,V(ref,x)*10k))
B2 gate   0      V = max(0,min(5,V(err,clock)*1k))
V3 +V     0      5.0
L1 +V     lx     220u
RL lx     out1   125m
M1 out1   gate   0  IRF510
D1 out1   OUT    MBRS340
C1 OUT    cx     33u
RC cx     0      50m
R4 out2   OUT    R = (time<12ms ? {Rload} : time<20ms ? {Rload/2} : {2*Rload})
V4 out2   0      0.0

.param Rload=100

.model IRF510 VDMOS nchan
+ Vto=3.6 Kp=1.3 Lambda=.001 Theta=0.07 ksubthres=.1
+ Rg=3 Rd=200m Rs=54m Rds=1e7
+ Cgdmax=.2n Cgdmin=.05n a=0.3 Cgs=.12n 
+ Is=17p N=1.1 Rb=80m XTI=3
+ Cjo=.25n Vj=0.8 m=0.5
+ tcvth=0.007 MU=-1.27 texp0=1.5

.model MBRS340 D(Is=22.6u Rs=.042 N=1.094 Cjo=480p M=.61 Eg=.69 Xti=2)

.control
   listing e
   option method=gear
   tran 10n 30m 0 5n
*   write dcdc.raw
   plot v(err) v(clock) v(gate) v(out)
   plot -i(V3) i(V4) ylimit 0 1
   rusage all
.endc

.end
