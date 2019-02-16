VDMOS output

m1 d g s IRFZ48Z

.model IRFZ48Z VDMOS (Rg = 1.77 Vto=4 Rd=1.85m Rs=0.0m Rb=3.75m Kp=25 Cgdmax=2.1n Cgdmin=0.05n Cgs=1.8n Cjo=0.55n Is=2.5p tt=20n ksubthres=0.1 mfg=International_Rectifier Vds=55 Ron=8.6m Qg=43n)

vd d 0 1
vg g 0 1
vs s 0 0

.control
dc vd -1 15 0.05 vg 3 7 1
plot vs#branch
dc vg 2 7 0.05 vd 0.5 2.5 0.5
plot vs#branch ylog
.endc

.end
