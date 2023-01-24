***  Single NMOS and PMOS Transistor BSIM4 (Id-Vgs, Vbs) (Id-Vds, Vgs) (Id-Vgs, T)  ***

M1 2 1 3 4 n1 W=1u L=0.35u Pd=1.5u Ps=1.5u ad=1.5p as=1.5p
vgsn 1 0 3.5
vdsn 2 0 0.1
vssn 3 0 0
vbsn 4 0 0

M2 22 11 33 44 p1 W=2.5u L=0.35u Pd=3u Ps=3u ad=2.5p as=2.5p
vgsp 11 0 -3.5
vdsp 22 0 -0.1
vssp 33 0 0
vbsp 44 0 0

* modified parameters
*.model n1 nmos level=49 version=3.3.0 tox=10n nch=1e17 nsub=5e16
*.model p1 pmos level=49 version=3.3.0 tox=10n nch=1e17 nsub=5e16

* BSIM3v3.3.0 model with internal parameters only
.model n1 nmos level=54 version=4.8.2
.model p1 pmos level=54 version=4.8.2

.control
set xgridwidth=2
set xbrushwidth=3

* NMOS
dc vgsn 0 1.5 0.05 vbsn 0 -2.5 -0.5
plot vssn#branch ylabel 'Id vs. Vgs, Vbs 0 ... -2.5'
plot abs(vssn#branch) ylog ylabel 'Id vs. Vgs, Vbs 0 ... -2.5'
dc vdsn 0 2 0.05 vgsn 0 2 0.4
plot vssn#branch ylabel 'Id vs. Vds, Vgs 0 ... 2'
dc vgsn 0 1.5 0.05 temp -40 160 40
plot vssn#branch ylabel 'Id vs. Vds, Temp. -40 ... 160'
plot abs(vssn#branch) ylog ylabel 'Id vs. Vds, Temp. -40 ... 160'

* PMOS
dc vgsp 0 -1.5 -0.05 vbsp 0 2.5 0.5
plot vssp#branch ylabel 'Id vs. Vgs, Vbs 0 ... 2.5'
plot abs(vssp#branch) ylog ylabel 'Id vs. Vgs, Vbs 0 ... 2.5'
dc vdsp 0 -2 -0.05 vgsp 0 -2 -0.4
plot vssp#branch ylabel 'Id vs. Vds, Vgs 0 ... -2'
dc vgsp 0 -1.5 -0.05 temp -40 160 40
plot vssp#branch ylabel 'Id vs. Vds, Temp. -40 ... 160'
plot abs(vssp#branch) ylog ylabel 'Id vs. Vds, Temp. -40 ... 160'
.endc

.end
