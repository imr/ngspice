VDMOS p channel output

m1 d g s IRF7233
.model IRF7233 VDMOS(pchan Rg=3 Rd=8m Rs=6m Vto=-1 Kp=70 Cgdmax=2n Cgdmin=.25n Cgs=3.3n Cjo=.98n Is=98p Rb=10m mfg=International_Rectifier Vds=-12 Ron=20m Qg=49n)

m2 d g s2 IRF7233_2
.model IRF7233_2 VDMOS(pchan mtriode=2 Rg=3 Rd=8m Rs=6m Vto=-1 Kp=70 Cgdmax=2n Cgdmin=.25n Cgs=3.3n Cjo=.98n Is=98p Rb=10m mfg=International_Rectifier Vds=-12 Ron=20m Qg=49n)

m3 d g s3 IRF7233_3
.model IRF7233_3 VDMOS(pchan mtriode=2 Rg=3 Rd=8m Rs=6m Vto=-1 Kp=70 Cgdmax=2n Cgdmin=.25n Cgs=3.3n Cjo=.98n Is=98p Rb=10m mfg=International_Rectifier Vds=-12 Ron=20m Qg=49n ksubthres=0.1)

vd d 0 -5
vg g 0 -5
vs s 0 0
vs2 s2 0 0
vs3 s3 0 0

.control
dc vd -12 1 0.05 vg 0 -5 -1
plot vs#branch vs2#branch vs3#branch
dc vg 0 -4 -0.05 vd -1 -12 -2
plot vs#branch vs2#branch vs3#branch
plot log(-vs#branch) log(-vs2#branch) log(-vs3#branch)
.endc

.end
