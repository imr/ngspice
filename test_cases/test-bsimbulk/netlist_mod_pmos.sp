OSDI BSIMBULK PMOS Test
*.options abstol=1e-15

* one voltage source per MOS terminal:
VD dd 0 -1
VG gg 0 -1
VS ss 0 0
VB bb 0 0

* model definitions:
*.model BSIMBULK_osdi_P bsimbulk type=-1
.include Modelcards/model.l

*OSDI BSIMBULK:
*
A1 dd gg ss bb BSIMBULK_osdi_P W=5e-6 L=5e-7

.control
pre_osdi test_osdi_win/bsimbulk106.osdi
set xbrushwidth=3
* a DC sweep: drain, gate
*op
dc Vd -1.8 0 0.01 VG -0.2 -1.8 -0.2 ; o.k.
*dc Vd 0 -1.8 -0.01 VG -0.2 -1.8 -0.2 ; not o.k.
* plot source current
plot i(VS)

.endc

.end
