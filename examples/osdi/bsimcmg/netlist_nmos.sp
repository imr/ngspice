OSDI BSIMCMG Test
*.options abstol=1e-15

* one voltage source per MOS terminal:
VD dd 0 1
VG gg 0 1
VS ss 0 0
VB bb 0 0

* model definitions:
*.model bsim4_osdi bsim4va
.include Modelcards/modelcard.nmos

*OSDI BSIM4:
* Where to put instance parameters channel width and length?
N1 dd gg ss bb BSIMCMG_osdi_N ; W=5u L=0.2u

.control
* pre_osdi ../osdi_libs/bsimcmg.osdi
set xbrushwidth=3
* a DC sweep: drain, gate
dc Vd 0 2.5 0.01 VG 0 2.5 0.5
* plot source current
plot i(VS)

.endc

.end
