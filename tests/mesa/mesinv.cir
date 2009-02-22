* Mesfet Inverter with ungated load/Wload=2e-6
* Taken from macspice3f4 package
*
* Output node is 70
*

bl1 10 40 i=0.0005*tanh(v(10,40)/0.0005/1120)*(1+v(10,40)*0.027)
bl2 20 50 i=0.0005*tanh(v(20,50)/0.0005/1120)*(1+v(20,50)*0.027)

zd1 70 2 0 driver l=0.7u w=20u
zd2 80 70 0 driver l=0.7u w=20u

rdl1 1 10 20
rdl2 1 20 20
rsl1 40 70 20
rsl2 50 80 20

vin 1000 0 dc 0
vdd 2000 0 dc 1.6
visrc 2000 1 dc 0
vig  1000 2 dc 0

.model driver nmf
+ level=2
+ n=1.44
+ rd=20
+ rs=20
+ ri=10
+ rf=10
+ vs=1.9e5
+ mu=0.25
+ d=1e-7
+ vto=0.15
+ m=2
+ lambda=0.15
+ sigma0=0.02
+ vsigmat=0.5

*.nodeset v(10)=1.6 v(40)=1.6 v(70)=1.6 v(20)=0.2 v(50)=0.2
*+ v(80)=0.2

.options noacct

.dc vin 0 1 0.01

.print dc v(70)

.end

