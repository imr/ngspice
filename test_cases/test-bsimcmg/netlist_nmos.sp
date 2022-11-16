OSDI BSIMCMG Test
*.options abstol=1e-15

* one voltage source per MOS terminal:
VD dd 0 1
VG gg 0 1
VS ss 0 0
VB bb 0 0

* model definitions:
*.model BSIMCMG_osdi_N
.include Modelcards/modelcard.nmos

*OSDI BSIMCMG:
* Fin thickness, Designed gate length, Number of fins per finger,
* Number of source diffusion squares, Number of drain diffusion squares
N1 dd gg ss bb BSIMCMG_osdi_N TFIN=15n L=30n NFIN=10 NRS=1 NRD=1

.control
pre_osdi test_osdi_win/bsimcmg.osdi
set xbrushwidth=3
* a DC sweep: drain, gate
dc Vd 0 2.5 0.01 VG 0 2.5 0.5
* plot source current
plot i(VS)

.endc

.end
