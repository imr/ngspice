*Sample netlist for BSIM-MG
*Id-Vd Characteristics for NMOS (T = 27 C)

.option abstol=1e-6 reltol=1e-6 post ingold
.temp -55

.hdl "bsimcmg.va"
.include "modelcard.nmos.1"

* --- Voltage Sources ---
vds drain  0 dc=0
vgs gate  0 dc=1.0
vbs bulk  0 dc=0.2

* --- Transistor ---
X1 drain gate 0 bulk nmos1 TFIN=15n L=40n NFIN=10 NRS=1 NRD=1

* --- DC Analysis ---
.dc vds 0 1 0.01 vgs 0 1.0 0.1
.probe dc ids=par`-i(vds)`
.probe dc gds=deriv(ids)
.print dc par'ids' par'gds'

.alter
.temp 27

.alter
.temp 100

.end
