VDMOS output

*m1 d g s IRFZ48Z
m1 d g s SQ7002K

m2 d g s2 SQ7002K_2

*.model IRFZ48Z VDMOS (Rg = 1.77 Vto=4 Rd=1.85m Rs=0.0m Rb=3.75m Kp=25 Cgdmax=2.1n Cgdmin=0.05n Cgs=1.8n Cjo=0.55n Is=2.5p tt=20n mfg=International_Rectifier Vds=55 Ron=8.6m Qg=43n)

.MODEL SQ7002K VDMOS(KP=0.46 RS=0.8751 RG=150 VTO=1.8 rds=50Meg LAMBDA=60m CGDMAX=20p CGDMIN=2p CGS=17p TT=500n a=0.47 IS=3.25n N=1.744 RB=0.118608 m=0.348 Vj=0.23 Cjo=14pF mtriode=1 Vds=60 Ron=1 Qg=0.9n mfg=VISHAY)

.MODEL SQ7002K_2 VDMOS(KP=0.46 RS=0.8751 RG=150 VTO=1.8 rds=50Meg LAMBDA=60m CGDMAX=20p CGDMIN=2p CGS=17p TT=500n a=0.47 IS=3.25n N=1.744 RB=0.118608 m=0.348 Vj=0.23 Cjo=14pF mtriode=2 Vds=60 Ron=1 Qg=0.9n mfg=VISHAY)

vd d 0 1
vg g 0 1
vs s 0 0
vs2 s2 0 0

.dc vd -1 7 0.05 vg 3 7 1

.control
run
plot vs#branch vs2#branch
.endc

.end
