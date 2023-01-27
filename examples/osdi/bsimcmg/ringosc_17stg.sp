* Sample netlist for BSIM-CMG
* 17-stage ring oscillator

.include Modelcards/modelcard.nmos
.include Modelcards/modelcard.pmos

* --- Voltage Sources ---
vdd   supply  0 dc=1.0

* --- Inverter Subcircuit ---
.subckt mg_inv vin vout vdd gnd
NP1 vout vin vdd vdd BSIMCMG_osdi_P
NN1 vout vin gnd gnd BSIMCMG_osdi_N
.ends

* --- 17 Stage Ring oscillator ---
Xinv1   1  2 supply 0 mg_inv
Xinv2   2  3 supply 0 mg_inv
Xinv3   3  4 supply 0 mg_inv
Xinv4   4  5 supply 0 mg_inv
Xinv5   5  6 supply 0 mg_inv
Xinv6   6  7 supply 0 mg_inv
Xinv7   7  8 supply 0 mg_inv
Xinv8   8  9 supply 0 mg_inv
Xinv9   9 10 supply 0 mg_inv
Xinv10 10 11 supply 0 mg_inv
Xinv11 11 12 supply 0 mg_inv
Xinv12 12 13 supply 0 mg_inv
Xinv13 13 14 supply 0 mg_inv
Xinv14 14 15 supply 0 mg_inv
Xinv15 15 16 supply 0 mg_inv
Xinv16 16 17 supply 0 mg_inv
Xinv17 17  1 supply 0 mg_inv

* --- Initial Condition ---
.ic  v(1)=1

.tran 1p 10n

* measure oscillator frequency by period delta t
.measure tran t1 when v(1)=0.5 cross=1
.measure tran t2 when v(1)=0.5 cross=7
.measure tran period param='(t2-t1)/3'
.measure tran frequency param='3/(t2-t1)'
.measure tran delay_per_stage param='period/34'

.control
* pre_osdi ../osdi_libs/bsimcmg.osdi
set xbrushwidth=3
run
rusage time
plot v(1) xlimit 1n 2n
* measure oscillator frequency with fft
fft V(1)
let magv1 = mag(v(1))
plot  magv1 xlimit 0 16G
meas sp fmax max magv1 from=10G to=20G
.endc

.end
