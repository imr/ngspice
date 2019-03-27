***** VBIC99 level 9 Transient test *****
*
q1 3 2 0 0 t vbic99
v 4 0 dc 5.0
vin 1 0 dc 2.5 pulse (0 5 0 1n 1n 10n 25n)
r1 1 2 100
r2 3 4 10k
*
.control
op
tran 10p 600n
plot v(1) v(2) v(3) v(4)
plot v(t)
.endc
*
.include bjt_vbic.mod
*
.end

