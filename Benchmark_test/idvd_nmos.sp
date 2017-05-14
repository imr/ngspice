*Sample netlist for BSIM6.0
*Id-Vd Characteristics for NMOS (T = 27 C)

.option abstol=1e-6 reltol=1e-6 post ingold
.temp 27
.hdl "bsim6.va"
.include "modelcard.nmos"

* --- Voltage Sources ---
vd d  0 dc=1.3
vg g  0 dc=0
vs s  0 dc=0
vb b  0 dc=0

* --- Transistor ---
X1 d g s b nmos W=10e-6 L=10e-6

* --- DC Analysis ---
.dc  vd 0.0 1.3 0.01 vg 0.4 1 0.3
.probe dc ids=par'-i(vd)'
.probe dc gd=deriv(ids)
.print dc par'ids' par'gd' 
.end
