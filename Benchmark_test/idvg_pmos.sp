*Sample netlist for BSIM6.0
.option abstol=1e-6 reltol=1e-6 post ingold

.hdl "bsim6.va"
.include "modelcard.pmos"


* --- Voltage Sources ---
vd d  0 dc=-0.05
vg g  0 dc=0.0
vs s  0 dc=0.0
vb b  0 dc=0.0

* --- Transistor ---
X1 d g s b pmos W=10e-6 L=10e-6

* --- DC Analysis ---
.dc  vg -1.3.0 1.3 0.01 vb 0 -0.3 -0.1
.probe dc ids=par'i(vd)' 
.probe dc gm=deriv(ids)
.probe dc gm2= deriv(gm)
.print dc par'ids' par'gm' par'gm2'
.end

