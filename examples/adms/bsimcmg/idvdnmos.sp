*Sample netlist for BSIM-MG
* (exec-spice "ngspice %s" t)
*Id-Vd Characteristics for NMOS (T = 27 C)

.option abstol=1e-6 reltol=1e-6 post ingold

*.hdl "bsimcmg.va"
.include "modelcard.nmos.1"

* --- Voltage Sources ---
vds drain  0 dc=0
vgs gate  0 dc=1.0
vbs bulk  0 dc=0.2

* --- Transistor ---
m1 drain gate 0 bulk 0 nmos1 TFIN=15n L=40n NFIN=10 NRS=1 NRD=1 D=40n

* --- DC Analysis ---
.dc vds -0.1 1 0.01 vgs 0 1.0 0.1
*.probe dc ids=par`-i(vds)`
*.probe dc gds=deriv(ids)
*.print dc par'ids' par'gds'

.control
save @m1[gds]
save vds#branch

set temp = -55
run
let ids = -i(vds)
let xgds = deriv(ids)
plot ids
plot xgds
plot @m1[gds]

set temp = 27
run
let ids = -i(vds)
let xgds = deriv(ids)
plot ids
plot xgds
plot @m1[gds]

set temp = 100
run
let ids = -i(vds)
let xgds = deriv(ids)
plot ids
plot xgds
plot @m1[gds]

*show all

.endc

.end
