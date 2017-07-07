*Samle netlist for BSIM-MG
* Drain Noise Simulation 

.option abstol=1e-6 reltol=1e-6 post ingold
.temp 27

.hdl "bsimcmg.va"
.include "modelcard.nmos"

* --- Voltage Sources ---
vds 1 0 dc=1v
vgs gate 0 dc=0.5v ac=1
vbs bulk 0 dc=0v

* --- Circuit ---
lbias 1 drain 1m
cload drain 2 1m
rload 2 0 R=1 noise=0
X1 drain gate 0 bulk nmos1 TFIN=15n L=30n NFIN=10 NRS=1 NRD=1

* --- Analysis ---
.op
*.dc vgs -0.5 1.5 0.01
*.print dc i(lbias)
.ac dec 11 1k 100g
.noise v(drain) vgs 1
*.print ac i(cload)
.print ac v(drain)
.print noise inoise onoise
.end

