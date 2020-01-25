*****************==== Inverter ====*******************
*********** VDMOS ****************************
vdd 1 0 5
vss 4 0 0

.subckt inv out in vdd vss
mp1 out in vdd p1
mn1 out in vss n1
.ends

xinv 3 2 1 4 inv

Vin 2 0 DC 0 Pulse (0 5 10n 10n 10n 140n 300n)

.control
dc Vin 0 5 0.05
* current and output in a single plot
plot v(2) v(3) vss#branch
tran 1n 1u
* current and output in a single plot
plot v(2) v(3)
.endc

.model  N1  vdmos cgdmin=0.2p cgdmax=1p a=2 cgs=0.5p rg=5k rb=1e9 cjo=0.1p
.model  P1  vdmos cgdmin=0.2p cgdmax=1p a=2 cgs=0.5p rg=5k rb=1e9 cjo=0.1p pchan
.end
