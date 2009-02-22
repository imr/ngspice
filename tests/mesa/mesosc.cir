* Mesfet Ring Oscillator with ungated load
* Taken form macspice3f4
*

.subckt mesinv 10 20 3
* Node 10: Power Supply
* Node 20: Input
* Node 30: Output
rdl 10 1 20
bl1 1 2 i=0.00025*tanh(v(1,2)/0.00025/2240)*(1+v(1,2)*0.027)
rsl 2 3 20
zd 3 20 0 driver l=0.7u w=20u
ci 3 0 20f
.ends mesinv

.options noacct
.model driver nmf level=2 n=1.44 rd=20 rs=20 vs=1.9e5
+ mu=0.25 d=1e-7 vto=0.15 m=2 lambda=0.15 sigma0=0.02
+ vsigmat=0.5

vdd 1 0 dc 1.6

xinv01 1 2 3 mesinv
xinv02 1 3 4 mesinv
xinv03 1 4 5 mesinv
xinv04 1 5 6 mesinv
xinv05 1 6 7 mesinv
xinv06 1 7 8 mesinv
xinv07 1 8 9 mesinv
xinv08 1 9 10 mesinv
xinv09 1 10 11 mesinv
xinv10 1 11 12 mesinv
xinv11 1 12 2000 mesinv

vnoise 2000 2 dc 0 pwl(0 0 0.2n 0 0.3n 0.1 0.4n 0)

.tran 1p 5n 1n 20p

.print tran V(8)

.end
