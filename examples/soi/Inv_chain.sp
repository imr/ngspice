Mx  Drain in Source Back-gate(substrate) Body  Tx  W  L (body ommitted for FB)

.include ./bsim4soi/nmos4p0.mod
.include ./bsim4soi/pmos4p0.mod
*.option TEMP=27C ITL4=100 RELTOL=.01 GMINSTEPS=200 ABSTOL=1N VNTOL=1M

Vpower VD 0 1.5
Vgnd VS 0 0

Vgate   in   VS PULSE(0v 1.5v 100ps 50ps 50ps 200ps 500ps)

*drain gate source substrate body contact
MN0 out0 in VS VS VS N1 W=5u L=0.18u
MP0 out0 in VD VS VD P1 W=10u L=0.18u
MN1 out1 Out0 VS VS VS N1 W=5u L=0.18u
MP1 out1 Out0 VD VS VD P1 W=10u L=0.18u
MN2 out2 Out1 VS VS VS N1 W=5u L=0.18u
MP2 out2 Out1 VD VS VD P1 W=10u L=0.18u
MN3 out3 Out2 VS VS VS N1 W=5u L=0.18u
MP3 out3 Out2 VD VS VD P1 W=10u L=0.18u
MN4 out4 Out3 VS VS VS N1 W=5u L=0.18u
MP4 out4 Out3 VD VS VD P1 W=10u L=0.18u

.tran 6p 600p
.print tran v(in) v(out4)

.control
if $?batchmode
* do nothing
else
  run
  plot in out4
endif
.endc

.END
