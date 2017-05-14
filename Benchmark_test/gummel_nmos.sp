*Sample netlist for BSIM6.0
*Drain current symmetry for nmos
.option abstol=1e-6 reltol=1e-6 post ingold

.hdl "bsim6.va"
.include "modelcard.nmos"

* --- Voltage Sources ---
vdrain drain 0 dc=0
esource source 0 drain 0 -1
vgate gate  0 dc=1.0
vbulk bulk 0 dc=0.0


* --- Transistor ---
X1 drain gate source bulk bsim6 W=10u L=10u 

* --- DC Analysis ---
.dc vdrain -0.1 0.1 0.001 vgate 0.0 1.0 0.2
.probe dc ids=par'-i(vdrain)'
.probe dc gx=deriv(ids)
.probe dc gx2=deriv(gx)
.probe dc gx3=deriv(gx2)
.probe dc gx4=deriv(gx3)
.print dc par'ids' par'gx' par'gx2' par'gx3' par 'gx4'

.end
