 Mx  Drain Gate Source Back-gate(substrate) Body  Tx  W  L (body ommitted for FB)

.include ./bsim4soi/nmos4p0.mod
.include ./bsim4soi/pmos4p0.mod
.option TEMP=27C

Vpower VD 0 1.5
Vgnd VS 0 0
Vgate Gate 0 0.0
MN0 Out Gate VS VS VS N1 W=10u L=0.18u
MP0 Out Gate VD VS VD P1 W=20u L=0.18u
.dc Vgate 0 1.5 0.05
.print dc v(out)

.control
if $?batchmode
* do nothing
else
  run
  plot out
  plot Vgnd#branch
endif
.endc

.END
