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
N1 dd gg ss bb BSIMBULK_osdi_N W=2000n L=500n
*N1 dd gg ss bb BSIMBULK_osdi_N W=200n L=50n ; seg fault in descr->setup_instance()

.control
pre_osdi test_osdi_win/bsimbulk107.osdi
set xbrushwidth=3
* a DC sweep: drain, gate
dc Vd 1.6 0 -0.01 VG 0.2 1.6 0.2
*dc Vd 0 1.6 0.01 VG 0.2 1.6 0.2 ; first dc point fails, not o.k.
* plot source current
plot i(VS)

.endc

.end
