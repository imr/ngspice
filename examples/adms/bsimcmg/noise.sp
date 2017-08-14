*Samle netlist for BSIM-MG
* (exec-spice "ngspice %s" t)
* Drain Noise Simulation 

.option abstol=1e-6 reltol=1e-6 post ingold
.temp 27

*.hdl "bsimcmg.va"
.include "modelcard.nmos"

* --- Voltage Sources ---
vds 1 0 dc=1v
vgs gate 0 dc=0.5v ac=1
vbs bulk 0 dc=0v

* --- Circuit ---
lbias 1 drain 1m
cload drain 2 1m
rload 2 0 R=1 noise=0
M1 drain gate 0 bulk 0 nmos1 TFIN=15n L=30n NFIN=10 NRS=1 NRD=1
+ FPITCH  = 4.00E-08

* --- Analysis ---
*.op
**.dc vgs -0.5 1.5 0.01
**.print dc i(lbias)
*.ac dec 11 1k 100g
*.noise v(drain) vgs 1
**.print ac i(cload)
*.print ac v(drain)
*.print noise inoise onoise

.control
op

ac dec 11 1k 100g
plot vdb(drain)

noise v(drain) vgs dec 11 1k 100g
print all
echo "silence in the studio, no noise today"

.endc

.end

