SOI Inverter
* Mx  Drain Gate Source Back-gate(substrate) Body  Tx  W  L (body ommitted for FB)

.include ./bsim4soi/nmos4p0.mod
.include ./bsim4soi/pmos4p0.mod
.option TEMP=27C

Vpower VD 0 1.5
Vgnd VS 0 0

Vgate   Gate   VS DC 0 PULSE(0v 1.5v 100ps 50ps 50ps 200ps 500ps)

*MN0 Out Gate VS VS VS N1 W=10u L=0.18u debug=1
*MP0 Out Gate VD VS VD P1 W=20u L=0.18u debug=1

MN0 Out Gate VS VS N1 W=10u L=0.18u Pd=11u Ps=11u
MP0 Out Gate VD VS P1 W=20u L=0.18u Pd=11u Ps=11u

.tran 3p 600ps
.print tran v(gate) v(out)

.control
if $?batchmode
* do nothing
else
  run
  plot Vgnd#branch
  plot gate out
endif
.endc

.END
