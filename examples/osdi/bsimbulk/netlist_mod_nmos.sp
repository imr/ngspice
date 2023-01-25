OSDI BSIMBULK NMOS Test
*.options abstol=1e-15

* one voltage source per MOS terminal:
VD dd 0 1
VG gg 0 1
VS ss 0 0
VB bb 0 0

* model definitions:
*.model BSIMBULK_osdi_N bsimbulk type=1
.include Modelcards/model.l

*OSDI BSIMBULK:
N1 dd gg ss bb BSIMBULK_osdi_N W=500n L=90n

.control
* pre_osdi ../osdi_libs/bsimbulk107.osdi
set xbrushwidth=3
* a DC sweep: drain, gate
dc Vd 0 1.6 0.01 VG 0 1.6 0.2
* plot source current
plot i(VS)

.endc

.end
