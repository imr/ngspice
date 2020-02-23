*****Single NMOS and PMOS Transistor For BSIM3 threshold voltage check (Id-Vgs) (Id-Vds) ***

M1 2 1 3 4 n1 W=1u L=0.35u Pd=1.5u Ps=1.5u ad=1.5p as=1.5p
vgs 1 0 3.5 
vds 2 0 0.1 
vss 3 0 0
vbs 4 0 0

M2 22 11 33 44 p1 W=2.5u L=0.35u Pd=3u Ps=3u ad=2.5p as=2.5p
vgsp 11 0 -3.5 
vdsp 22 0 -0.1 
vssp 33 0 0
vbsp 44 0 0

.options Temp=27.0

* BSIM3v3.3.0 model with modified default parameters 0.18µm
.model n1 nmos level=49 version=3.3.0 tox=3.5n nch=2.4e17 nsub=5e16 vth0=0.15
.model p1 pmos level=49 version=3.3.0 tox=3.5n nch=2.5e17 nsub=5e16 vth0=-0.15

*.include ./Modelcards/modelcard.nmos $ Berkeley model cards limited to L >= 0.35µm
*.include ./Modelcards/modelcard.pmos $ Berkeley model cards limited to L >= 0.35µm

* update of the default parameters required
*.model n1 NMOS level=49 version=3.3.0 $ nearly no current due to VT > 2 V ?
*.model p1 PMOS level=49 version=3.3.0

.control
set xgridwidth=2
set xbrushwidth=3
dc vgs 0 1.5 0.05 vbs 0 -2.5 -0.5
plot vss#branch
dc vds 0 2 0.05 vgs 0 2 0.4
plot vss#branch
dc vgsp 0 -1.5 -0.05 vbsp 0 2.5 0.5
plot vssp#branch
dc vdsp 0 -2 -0.05 vgsp 0 -2 -0.4
plot vssp#branch
.endc

.end





