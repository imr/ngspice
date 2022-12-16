OSDI EKV 2.6 PMOS Test
*.options abstol=1e-15

* one voltage source per MOS terminal:
VD dd 0 -1
VG gg 0 -1
VS ss 0 0
VB bb 0 0

* model definitions:
*.model .MODEL PCH EKV_VA type=-1
.include Modelcards/ekv26_0u5.par

*OSDI EKV:
N1 dd gg ss bb pch W=5e-6 L=5e-7

.control
pre_osdi test_osdi_win/ekv26_mod.osdi
set xbrushwidth=3
* a DC sweep: drain, gate
*op
dc Vd 0 -1.8 -0.01 VG 0 -1.8 -0.2
* plot source current
plot i(VS)

.endc

.end
