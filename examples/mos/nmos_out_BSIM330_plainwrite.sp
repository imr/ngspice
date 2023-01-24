*****  NMOS Transistor BSIM3 (Id-Vds) with Rd ***
** Node name with offensive character '+'

M1 2 1 3 4 n1 W=1u L=0.35u Pd=1.5u Ps=1.5u ad=1.5p as=1.5p
vgs 1 0 3.5 
vds 2 0 0.1 
vss 3 0 0
vbs 4 0 0

* drain series resistor
R2 2 22 1k
M2 22 1 +32 4 n1 W=1u L=0.35u Pd=1.5u Ps=1.5u ad=1.5p as=1.5p
vss2 +32 0 0


.options Temp=27.0

* BSIM3v3.3.0 model with modified default parameters 0.18µm
.model n1 nmos level=49 version=3.3.0 tox=3.5n nch=2.4e17 nsub=5e16 vth0=0.15
.model p1 pmos level=49 version=3.3.0 tox=3.5n nch=2.5e17 nsub=5e16 vth0=-0.15

.control
set xgridwidth=2
set xbrushwidth=3
dc vds 0 2 0.05 vgs 0 2 0.4
let v(/22) = V(22) ; only availavle in plot dc1
dc vds 0 2 0.05 vgs 0 2 0.5

let v(+22) = V(22) ; only availavle in plot dc2
set nolegend
*set plainplot
plot v(+22) plainplot
set plainwrite
*write test.out v(+22) vss#branch dc1.v(/22) dc1.vss#branch
unset nolegend
set color0=white
*unset plainplot ; required if 'set plainplot'
plot vss2#branch vs v(22) title 'Series resistor: Drain current versus drain voltage' xlabel 'Drain voltage' ylabel 'Drain current'
.endc

.end





