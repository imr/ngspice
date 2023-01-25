***  NMOS and PMOS transistors BSIMBULK (Id-Vgs, Vbs) (Id-Vds, Vgs) (Id-Vgs, T)  ***

Nn1 2 1 3 4 BSIMBULK_osdi_N l=0.1u  w=0.5u as=0.131175p ad=0.131175p ps=1.52u   pd=1.52u
vgsn 1 0 3.5
vdsn 2 0 0.1
vssn 3 0 0
vbsn 4 0 0

Np1 22 11 33 44 BSIMBULK_osdi_P  l=0.1u  w=1u  as=0.26235p  ad=0.26235p  ps=2.51u   pd=2.51u
vgsp 11 0 -3.5
vdsp 22 0 -0.1
vssp 33 0 0
vbsp 44 0 0

* BSIMBULK modelparameters for BSIMBULK106, Berkeley
.include Modelcards/model.l

.control
* Load the models dynamically
* pre_osdi ../osdi_libs/bsimbulk107.osdi
set xgridwidth=2
set xbrushwidth=3

* NMOS
dc vgsn 0 1.5 0.05 vbsn 0 -1.5 -0.3
plot vssn#branch ylabel 'Id vs. Vgs, Vbs 0 ... -1.5'
plot abs(vssn#branch) ylog ylabel 'Id vs. Vgs, Vbs 0 ... -1.5'
dc vdsn 0 1.6 0.01 vgsn 0 1.6 0.2
plot vssn#branch ylabel 'Id vs. Vds, Vgs 0 ... 1.6'
dc vgsn 0 1.5 0.05 temp -40 160 40
plot vssn#branch ylabel 'Id vs. Vds, Temp. -40 ... 160'
plot abs(vssn#branch) ylog ylabel 'Id vs. Vds, Temp. -40 ... 160'

* PMOS
dc vgsp 0 -1.5 -0.05 vbsp 0 1.5 0.3
plot vssp#branch ylabel 'Id vs. Vgs, Vbs 0 ... 1.5'
plot abs(vssp#branch) ylog ylabel 'Id vs. Vgs, Vbs 0 ... 1.5'
dc vdsp 0 -1.6 -0.01 vgsp 0 -1.6 -0.2
plot vssp#branch ylabel 'Id vs. Vds, Vgs 0 ... -1.6'
dc vgsp 0 -1.5 -0.05 temp -40 160 40
plot vssp#branch ylabel 'Id vs. Vds, Temp. -40 ... 160'
plot abs(vssp#branch) ylog ylabel 'Id vs. Vds, Temp. -40 ... 160'
.endc

.end
