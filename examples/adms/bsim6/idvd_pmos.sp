*Sample netlist for BSIM6.0
*Id-Vd Characteristics for PMOS (T = 27 C)
* (exec-spice "ngspice %s" t)

.option abstol=1e-6 reltol=1e-6 post ingold
.temp 27
.include "modelcard.pmos"


* --- Voltage Sources ---
vd d  0 dc=-1
vg g  0 dc=0
vs s  0 dc=0
vb b  0 dc=0

* --- Transistor ---
M1 d g s b  mp W=10e-6 L=10e-6

* --- DC Analysis ---
.dc  vd -1.3 0.0 0.01 vg -1 -0.4 0.3

.probe dc ids=par'i(vd)'
.probe dc gd=deriv(ids)
.print dc par'ids' par'gd'
.control
run
plot i(vd)
.endc
.end

