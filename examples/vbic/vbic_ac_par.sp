Ask for small signal values of vbic model

.include Infineon_VBIC.lib

v1 1 0 dc 5.0
*rc 1 c 2k
vb b 0 dc 0.6
q1 1 b 0 M_BFP780

.control
save @q1[gm] @q1[go] @q1[gpi]
save @q1[cbe] @q1[cbc]
save @q1[qbe] @q1[qbc]
dc v1 0.0 5 0.01
plot @q1[gm] @q1[go] @q1[gpi]
plot @q1[cbe] @q1[cbc]
plot @q1[qbe] @q1[qbc]
.endc

.end
