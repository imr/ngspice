DCFL inverter circuit

.subckt  inv 1 2 3
*
*Vdd 1.0
*Vin 2 0
*Vout 3  0
z1 1 3 3   aload l=1u w=10u
z2 3 2 0   adrv l=1u w=10u
.ends

vdd  1 0 dc 2
vin  2 0 dc 0 pwl(0,0V   1ns,0V 1.005ns,1V 2ns,1V)
x1  1 2 3 inv
x2  1 3 4 inv

.options noacct
.tran 0.01n 3n
.print tran all
.model adrv nhfet level=5 rd=60 rs=60 m=2.57 lambda=0.17
+ vs=1.5e5 mu=0.385 vt0=0.3 eta=1.32 sigma0=0.04
+ vsigma=0.1 vsigmat=0.3 js1s=1e-12 js1d=1e-12
+ nmax=6e15
.model aload nhfet level=5 rd=60 rs=60 m=2.57 lambda=0.17
+ vs=1.5e5 mu=0.385 vt0=-0.3 eta=1.32 sigma0=0.04
+ vsigma=0.1 vsigmat=0.3 js1s=1e-12 js1d=1e-12
+ nmax=6e15

.end
