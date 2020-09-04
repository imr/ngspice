BJT Quasi-Saturation Output Test Ic=f(Vc,Ib)

.include Infineon_VBIC.lib

IB 0 B 100u
VC C 0 0.0
VS S 0 0.0
Q1 C B 0 S M_BFP780

.control
dc vc 0.0 2.0 0.01 ib 100u 1000u 100u
plot -i(vc)
.endc

.end
