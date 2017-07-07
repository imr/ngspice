*Sample netlist for BSIM-MG
* (exec-spice "ngspice %s" t)
*Id-Vg Characteristics for PMOS (T = 27 C)

.option abstol=1e-6 reltol=1e-6 post ingold
.temp -55

*.hdl "bsimcmg.va"
.include "modelcard.pmos.1"

* --- Voltage Sources ---
vds supply  0 dc=-1
vgs gate  0 dc=-1
vbs bulk  0 dc=0

* --- Transistor ---
m1 supply gate 0 bulk 0 pmos1 TFIN=15n L=30n NFIN=10 NRS=1 NRD=1

* --- DC Analysis ---
.dc vgs 0.5 -1.0 -0.01 
.probe dc ids=par`i(vds)`
.probe dc gds=deriv(ids)
.print dc par'ids' par'-gds'

.alter
.temp 27

.alter
.temp 100

.control
run
plot i(vds)
* fixme, second temperature, and nasty reset issues
.endc

.end
