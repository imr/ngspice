*Samle netlist for BSIM6.0
* (exec-spice "ngspice %s" t)
* Drain Noise Simulation 

.option abstol=1e-6 reltol=1e-6 post ingold
.temp 27

.include "modelcard.nmos"

* --- Voltage Sources ---
vds 1 0 dc=1v ac=1
vgs gate 0 dc=0.5v 
vbs bulk 0 dc=0v

* --- Circuit ---
lbias 1 drain 1m
cload drain 2 1m
rload 2 0 R=1 noise=0
M1 drain gate 0 bulk0  mn W  = 10e-6 L = 10e-6

* --- Analysis ---
.op
*.dc vgs -0.5 1.3 0.01
*.print dc i(lbias)
.ac dec 11 1k 100g
.noise v(drain) vgs 1
*.print ac v(drain)
*.print dc v(drain)
.print noise inoise onoise
.control
run
.endc
.end

