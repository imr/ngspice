*Sample netlist for BSIM6.0
* (exec-spice "ngspice %s" t)
*17-stage ring oscillator

.options abstol=1e-6 reltol=1e-6 post ingold dcon=1

.include "modelcard.nmos"
.include "modelcard.pmos"

* --- Voltage Sources ---
vdd supply  0 dc=1.0

* --- Inverter Subcircuit ---
.subckt inverter vin vout vdd gnd
    Mp1 vout vin vdd gnd 0  mp  W=10e-6 L=10e-6
    Mn1 vout vin gnd gnd 0  mn  W=10e-6 L=10e-6
.ends

* --- 17 Stage Ring oscillator ---
Xinv1   1  2 supply 0 inverter
Xinv2   2  3 supply 0 inverter
Xinv3   3  4 supply 0 inverter
Xinv4   4  5 supply 0 inverter
Xinv5   5  6 supply 0 inverter
Xinv6   6  7 supply 0 inverter
Xinv7   7  8 supply 0 inverter
Xinv8   8  9 supply 0 inverter
Xinv9   9 10 supply 0 inverter
Xinv10 10 11 supply 0 inverter
Xinv11 11 12 supply 0 inverter
Xinv12 12 13 supply 0 inverter
Xinv13 13 14 supply 0 inverter
Xinv14 14 15 supply 0 inverter
Xinv15 15 16 supply 0 inverter
Xinv16 16 17 supply 0 inverter
Xinv17 17  1 supply 0 inverter

* --- Initial Condition ---
.ic  1=1

.tran 1n 10u

.print tran v(1)

*.measure tran t1 when v(1)=0.5 cross=1
*.measure tran t2 when v(1)=0.5 cross=7
*.measure tran period param'(t2-t1)/3'
*.measure tran delay_per_stage param'period/34'
.control
run
plot v(1)
.endc

.end


