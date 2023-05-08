*****************==== 11-Stage CMOS RO ====*******************
*********** MOS1 or VDMOS ************************************
vdd 1 0 5.0

.subckt inv out in vdd vss
mp1 out in vdd p1 m=2
mn1 out in vss n1
c1 out vss 0.2p
.ends

xinv1 3 2 1 0 inv
xinv2 4 3 1 0 inv
xinv3 5 4 1 0 inv
xinv4 6 5 1 0 inv
xinv5 7 6 1 0 inv
xinv6 8 7 1 0 inv
xinv7 9 8 1 0 inv
xinv8 10 9 1 0 inv
xinv9 2 10 1 0 inv

.model  N1  vdmos vto=1 cgdmin=0.05p cgdmax=0.2p a=1.2 cgs=0.15p rg=10 kp=1e-5 rb=1e7 cjo=1n ksubthres=0.2
.model  P1  vdmos vto=-1  cgdmin=0.05p cgdmax=0.2p a=1.2 cgs=0.15p rg=10 kp=1e-5 rb=1e7 cjo=1n pchan ksubthres=0.2

.tran 0.1n 10u
.ic v(6)=2.5

.control

run
rusage
* current and output in a single plot
plot v(6) 50000*(-I(vdd)) ylimit -1 6
.endc

.end
