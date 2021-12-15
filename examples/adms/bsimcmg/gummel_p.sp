*Sample netlist for BSIM-MG
* (exec-spice "ngspice %s" t)
*Drain current symmetry

.option abstol=1e-6 reltol=1e-6 post ingold

*.hdl "bsimcmg.va"
.include "modelcard.pmos"

* --- Voltage Sources ---
vdrain drain 0 dc=0
esource source 0 drain 0 -1
vgate gate  0 dc=-1.0
vbulk bulk 0 dc=0


* --- Transistor ---
m1 drain gate source bulk 0 pmos1 TFIN=15n L=30n NFIN=10 NRS=1 NRD=1
+ FPITCH  = 4.00E-08

* --- DC Analysis ---
.dc vdrain -0.1 0.1 0.001 vgate 0.0 -1.0 -0.2
*.probe dc ids=par'-i(vdrain)'
*.probe dc gx=deriv(ids)
*.probe dc gx2=deriv(gx)
*.probe dc gx3=deriv(gx2)
*.probe dc gx4=deriv(gx3)
*.print dc par'ids' par'gx' par'gx2' par'gx3' par 'gx4'

.control
run
let ids = -i(vdrain)
let gx = deriv(ids)
let gx2 = deriv(gx)
let gx3 = deriv(gx2)
let gx4 = deriv(gx3)
plot ids
plot gx
plot gx2
plot gx3
plot gx4

.endc

.end
